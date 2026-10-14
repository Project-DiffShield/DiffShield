import json
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

import scripts.budget_check as bc
import scripts.prepare_v1_data as pv
import scripts.run_job as rj


def test_budget_check_stem_pairs_and_pairs_csv(tmp_path):
    clean, prot = tmp_path / "c", tmp_path / "p"
    clean.mkdir()
    prot.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        a = rng.integers(10, 240, (32, 32, 3)).astype(np.int64)
        Image.fromarray(a.astype(np.uint8)).save(clean / f"im{i}.jpg", quality=100)
        ca = np.array(Image.open(clean / f"im{i}.jpg")).astype(np.int64)
        d = rng.integers(-4, 5, ca.shape)
        d[0, 0, 0] = 4
        Image.fromarray(np.clip(ca + d, 0, 255).astype(np.uint8)).save(prot / f"im{i}.png")
    res = bc.check(bc.collect_pairs(clean, prot))
    assert res["n_pairs"] == 2 and res["global_max_abs_diff"] == 4 and res["inferred_eps_255"] == 4
    assert res["clean_files_are_jpeg"] == 2
    # pairs_csv mapping with different protected names
    (prot / "im0.png").rename(prot / "protected_face_001.png")
    (prot / "im1.png").rename(prot / "protected_face_002.png")
    csv = tmp_path / "pairs.csv"
    pd.DataFrame({"Original_Filename": ["im1.jpg", "im0.jpg"],
                  "Protected_Filename": ["protected_face_002.png", "protected_face_001.png"]}).to_csv(csv, index=False)
    res2 = bc.check(bc.collect_pairs(clean, prot, csv))
    assert res2["global_max_abs_diff"] == 4 and res2["n_pairs"] == 2
    assert bc.main(["--clean_dir", str(clean), "--protected_dir", str(prot), "--pairs_csv", str(csv),
                    "--out_dir", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "budget_check.json").read_text())["inferred_eps_255"] == 4


def _fake_celebahq(root, names):
    (root / "CelebA-HQ-img").mkdir(parents=True)
    (root / "CelebAMask-HQ-attribute-anno.txt").write_text("0\n")
    for n in names:
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            root / "CelebA-HQ-img" / n)


def test_prepare_v1_data_from_manifest(tmp_path, repo_root):
    allnames = [l.strip() for l in (repo_root / "data/splits/v1_eval70.txt").read_text().splitlines()]
    names = sorted(allnames, key=lambda n: (n[:-4], n))[:3]  # what --limit 3 selects
    _fake_celebahq(tmp_path / "cel", names)
    assert pv.main(["--from_manifest", "--celebahq_root", str(tmp_path / "cel"), "--out_dir", str(tmp_path / "out"),
                    "--limit", "3"]) == 0
    got = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert got == sorted(names)
    assert Image.open(tmp_path / "out" / names[0]).size == (512, 512)


def test_job_pipeline_end_to_end_tiny(tmp_path, repo_root):
    """J01 end to end (prepare -> sharded protect -> merge -> budget_check -> tables -> compare) with tiny models."""
    allnames = [l.strip() for l in (repo_root / "data/splits/v1_eval70.txt").read_text().splitlines()]
    names = sorted(allnames, key=lambda n: (n[:-4], n))[:3]  # what --limit 3 selects
    _fake_celebahq(tmp_path / "input" / "cel", names)
    # the job's prepare step auto-discovers the dataset under /kaggle/input; here we pass it via extra env instead
    jobs = rj.load_jobs(repo_root / "kaggle" / "jobs.yaml")
    job = rj.resolve_job(jobs, "J01b")
    assert job["steps"] == jobs["J01"]["steps"] and "J01" in jobs
    work = tmp_path / "work"
    # patch the prepare step to use the fake root
    for st in jobs["J01"]["steps"]:
        if st["name"] == "prepare_data":
            st["run"] += ["--celebahq_root", str(tmp_path / "input" / "cel")]
    ypath = tmp_path / "jobs.yaml"
    import yaml

    ypath.write_text(yaml.safe_dump({"jobs": jobs}))
    rc = rj.main(["--jobs", "J01", "--jobs_file", str(ypath), "--work", str(work), "--n_gpus", "0",
                  "--set", "n_images=3", "--set", "iters=2", "--set", "backend=tiny", "--set", "image_size=64",
                  "--set", "run_tests=false", "--set", 'extra_args=["--no_lpips","--device","cpu"]'])
    assert rc == 0
    rd = work / "J01" / "results" / "J01"
    assert (work / "J01" / "DONE").exists() and (work / "J01_outputs.zip").exists()
    df = pd.read_csv(rd / "metrics.csv")
    assert len(df) == 3
    assert (rd / "budget_check.json").exists() and (rd / "tables" / "invisibility.md").exists()
    assert (rd / "kaggle_session.json").exists()
    assert (rd / "compare_v1.json").exists()
    # second invocation skips (resumable job list)
    assert rj.main(["--jobs", "J01", "--jobs_file", str(ypath), "--work", str(work), "--n_gpus", "0"]) == 0
