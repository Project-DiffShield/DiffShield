"""kaggle/jobs.yaml sanity: every session resolves (dry run), and a miniature session runs end to end on CPU with
fake evaluation backends (macros, smoke splits, prepare_data, sharded protect, eval jobs file, merge)."""
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

import scripts.run_job as rj

SESSIONS = ["J00", "J01", "J01b"] + [f"S{i:02d}" for i in range(1, 19)]


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("job", SESSIONS)
def test_every_session_resolves_in_dry_run(job, smoke, repo_root, tmp_path, capsys):
    args = ["--jobs", job, "--work", str(tmp_path), "--dry_run", "--n_gpus", "2"]
    if smoke:
        args += ["--set", "smoke=true"]
    assert rj.main(args) == 0
    out = capsys.readouterr().out
    assert "UNRESOLVED" not in out, out


def test_step_names_unique_within_session(repo_root):
    jobs = rj.load_jobs(repo_root / "kaggle" / "jobs.yaml")
    for jid in SESSIONS:
        steps = rj.expand_steps(rj.resolve_job(jobs, jid)["steps"], jobs["_macros"])
        names = [s["name"] for s in steps]
        assert len(names) == len(set(names)), (jid, [n for n in names if names.count(n) > 1])
        for s in steps:
            assert "run" in s or "builtin" in s, (jid, s["name"])


def test_all_scripts_referenced_exist(repo_root):
    jobs = rj.load_jobs(repo_root / "kaggle" / "jobs.yaml")
    for jid in SESSIONS:
        for s in rj.expand_steps(rj.resolve_job(jobs, jid)["steps"], jobs["_macros"]):
            for a in s.get("run", []):
                if isinstance(a, str) and re.match(r"^scripts/\w+\.py$", a):
                    assert (repo_root / a).exists(), (jid, a)


def test_dependent_sessions_declare_prev_root(repo_root):
    jobs = rj.load_jobs(repo_root / "kaggle" / "jobs.yaml")
    for jid in ("S15", "S16"):
        assert "prev_root" in rj.resolve_job(jobs, jid)["params"]


def _fake_celebahq(root, names):
    (root / "CelebA-HQ-img").mkdir(parents=True)
    (root / "CelebAMask-HQ-attribute-anno.txt").write_text("0\n")
    rng = np.random.default_rng(1)
    for n in names:
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(root / "CelebA-HQ-img" / n)


def test_mini_session_end_to_end_with_fake_backends(tmp_path, repo_root, monkeypatch):
    names = [l.strip() for l in (repo_root / "data/splits/v1_eval70.txt").read_text().splitlines()][:10]
    cel = tmp_path / "input" / "cel"
    _fake_celebahq(cel, names)
    monkeypatch.setattr(rj, "find_celebahq", lambda root="/kaggle/input": cel)

    real = yaml.safe_load((repo_root / "kaggle" / "jobs.yaml").read_text())
    mini = {"macros": real["macros"], "jobs": {"MINI": {
        "params": {"smoke": True},
        "smoke_params": {"n_images": 3},
        "steps": [
            {"name": "resolve_splits", "builtin": "resolve_splits"},
            {"name": "make_config", "builtin": "make_config"},
            {"macro": "prep", "with": {"splits": ["test"]}},
            {"macro": "protect", "with": {"method": "noise_uniform", "split": "test",
                                          "extra": ["--no_lpips", "--device", "cpu"]}},
            {"macro": "evaluate", "with": {"name": "ev", "defs": {"backend": "fake", "device": "cpu"},
                                           "jobs": [{"name": "eval_noise_H", "tier": "H", "method": "noise_uniform",
                                                     "split": "test", "editor": "sd15_img2img,ipadapter_faceid"},
                                                    {"name": "eval_clean_H", "tier": "H", "method": "clean",
                                                     "split": "test", "editor": "sd15_img2img"}]}},
            {"name": "report", "run": ["{py}", "scripts/smoke_report.py", "--root", "{results}"], "soft": True},
        ]}}}
    jf = tmp_path / "jobs.yaml"
    jf.write_text(yaml.safe_dump(mini))
    work = tmp_path / "work"
    rc = rj.main(["--jobs", "MINI", "--jobs_file", str(jf), "--work", str(work), "--n_gpus", "0"])
    sess = json.loads((work / "MINI" / "kaggle_session.json").read_text())
    assert rc == 0 and sess["ok"], sess
    res = work / "MINI" / "results"
    prot = pd.read_csv(res / "prot_noise_uniform_test" / "metrics.csv")
    assert len(prot) == 3 and prot.png_maxabs_u8.max() <= 4
    ev = pd.read_csv(res / "eval_noise_H" / "metrics.csv")
    assert len(ev) > 0 and set(ev.method) == {"noise_uniform"} and set(ev.editor) <= {"sd15_img2img", "ipadapter_faceid"}
    assert (work / "MINI" / "DONE").exists() and (work / "MINI_outputs.zip").exists()
    # smoke mode used 1 edit seed
    assert ev.groupby(["image_id", "editor", "strength", "prompt_id"]).seed.nunique().max() == 1
