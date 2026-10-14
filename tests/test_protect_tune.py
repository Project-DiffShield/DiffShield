import importlib.util
import json
import os
import shutil

import pytest

pytest.importorskip("optuna")

import yaml
from PIL import Image

from test_protect_fakes import ROOT, TinySem, base_cfg, make_models

SPEC = importlib.util.spec_from_file_location("tune_optuna", os.path.join(ROOT, "scripts", "tune_optuna.py"))
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)


@pytest.fixture
def env(tmp_path):
    cfgdir = tmp_path / "config"
    (cfgdir / "parts").mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, "config", "parts", "protect.yaml"), cfgdir / "parts" / "protect.yaml")
    (cfgdir / "config.yaml").write_text(yaml.safe_dump(base_cfg(iters=2, k=1)))
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    names = []
    for i in range(4):
        Image.effect_noise((64, 64), 20).convert("RGB").save(imgs / f"v{i}.png")
        names.append(f"v{i}.png")
    (tmp_path / "val.txt").write_text("\n".join(names))
    T.PROTECTOR_COMPONENTS = {"models": make_models(), "sem": TinySem(),
                              "extra_vaes": {"sdxl_vae_fp16": make_models(seed=2).vae}}
    yield tmp_path
    T.PROTECTOR_COMPONENTS = None
    T.IMAGE_SIZE = 512


def args_for(p, n_trials, extra=()):
    return ["--config", str(p / "config" / "config.yaml"), "--storage", f"sqlite:///{p/'t.db'}", "--study", "s",
            "--n-trials", str(n_trials), "--n-images", "2", "--iters", "2", "--k-draws", "1", "--device", "cpu",
            "--val-list", str(p / "val.txt"), "--image-dir", str(p / "imgs"), "--image-size", "64",
            "--stub-score", "--startup", "2", "--out-dir", str(p / "out"), *extra]


def test_search_resume_and_constraints(env):
    import optuna

    T.main(args_for(env, 3))
    study = T.make_study(T.argparse.Namespace(storage=f"sqlite:///{env/'t.db'}", study="s", seed=1, startup=2))
    tr = study.get_trials()
    assert len(tr) == 3 and all(t.state.name == "COMPLETE" for t in tr)
    # trial 0 is the enqueued hand-set default
    assert tr[0].params["alpha"] == 0.27 and tr[0].params["antipure"] is False
    for t in tr:
        assert len(t.user_attrs["constraints"]) == 2 and "psnr" in t.user_attrs
        assert 0.1 <= t.params.get("ap_w", 0.5) <= 1.0 and 0 <= t.params["beta"] <= 1.5
    T.main(args_for(env, 3))          # resume: budget already used -> no new trials
    assert len(study.get_trials()) == 3
    T.main(args_for(env, 4))          # resume with larger budget -> exactly one more
    assert len(study.get_trials()) == 4


def test_confirm_stage_writes_candidate_or_reports(env):
    T.main(args_for(env, 3))
    T.main(args_for(env, 3, ["--confirm-top", "2", "--confirm-images", "2", "--confirm-iters", "2", "--confirm-k", "1"]))
    res = json.load(open(env / "out" / "s" / "confirm.json"))
    assert "hand_set" in res or "trial_0" in res
    assert all("psr" in v and "params" in v for v in res.values())


def test_params_to_override_and_constraints():
    ov = T.params_to_override({"alpha": 1, "eta": 2, "beta": 0.5, "gamma": 30, "antipure": True, "ap_w": 0.3}, 20, 2)
    assert ov["loss"]["anti_purify"] == {"enabled": True, "w_hf": 0.3, "w_mono": 0.3}
    assert ov["budget"]["iters"] == 20 and ov["eot"]["k_draws"] == 2
    assert T.constraints(40.0, 0.97) == [-2.0, -0.02000000000000002] or all(c < 0 for c in T.constraints(40.0, 0.97))
    assert T.constraints(37.0, 0.97)[0] > 0
