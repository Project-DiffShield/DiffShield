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


# ---- Step 1 (docs/STEP1_PLAN.md 2.3): v2next space, per-photo constraint, decision rule ----------------------------
def v2n_args(p, n_trials, extra=()):
    return args_for(p, n_trials, ["--space", "v2next", "--constraint", "per_photo", "--base-method", "v2_next", *extra])


def test_per_photo_constraint_math():
    assert T.constraints_per_photo([40.0, 37.5, 41.0], [0.99, 0.97, 0.99]) == pytest.approx([0.5, -0.02])
    assert all(c < 0 for c in T.constraints_per_photo([38.5, 39.0], [0.96, 0.99]))
    # the mean passes (39 / 0.97) but one photo fails -> only the per-photo form is infeasible
    assert all(c <= 0 for c in T.constraints(39.0, 0.97)) and T.constraints_per_photo([40.0, 38.0 - 1e-6], [0.97, 0.97])[0] > 0
    assert T.n_violations([40.0, 37.0, 39.0], [0.99, 0.99, 0.90]) == 2
    res = {"psnr": 39.0, "ssim": 0.97, "psnr_each": [40.0, 37.0], "ssim_each": [0.99, 0.95]}
    assert T.trial_constraints("mean", res) == T.constraints(39.0, 0.97)
    assert T.trial_constraints("per_photo", res)[0] == 1.0


def test_parse_editors_spec_and_n_images_zero(env):
    assert T.parse_editors(["sd15_img2img:0.55", "ipadapter_faceid"]) == [
        {"name": "sd15_img2img", "strength": 0.55}, {"name": "ipadapter_faceid"}]
    a = T.argparse.Namespace(edit_seeds=[1, 2], purifiers=["none", "vae_rt_sd15", "diffpure_t0.2"], editors=["sd15_img2img:0.55"])
    sp = T.build_spec(a)
    assert sp["seeds"] == [1, 2] and sp["purifiers"] == ["none", "vae_rt_sd15", "diffpure_t0.2"]
    assert sp["editors"] == [{"name": "sd15_img2img", "strength": 0.55}]
    old = T.build_spec(T.argparse.Namespace(edit_seeds=[1], purifiers=None, editors=None))
    assert old["purifiers"] == T.DEFAULT_SPEC["purifiers"] and old["editors"] == T.DEFAULT_SPEC["editors"]
    assert len(T.load_val_images(str(env / "val.txt"), str(env / "imgs"), 0, 64)) == 4
    assert len(T.load_val_images(str(env / "val.txt"), str(env / "imgs"), 3, 64)) == 3


def test_v2next_overrides_are_valid_and_resolve(env):
    from src.protect.v2_registry import deep_merge, resolve_cfg

    cfg = T.load_cfg(str(env / "config" / "config.yaml"))
    on = {"alpha": 1.0, "beta": 0.5, "gamma": 40.0, "den_on": True, "eta": 0.2, "den_t": "300-700", "p_onestep": 0.15}
    off = {"alpha": 1.0, "beta": 0.5, "gamma": 40.0, "den_on": False, "p_onestep": 0.0}      # trial.params omit eta/den_t
    ov = T.params_to_override(on, 20, 2)
    assert ov["p_onestep"] == 0.15 and ov["loss"]["den_t_range"] == [300, 700] and ov["loss"]["eta"] == 0.2
    assert ov["loss"]["lambda_id"] == 0.0 and ov["loss"]["anti_purify"]["enabled"] is False
    assert ov["budget"]["iters"] == 20 and ov["eot"]["k_draws"] == 2
    ov0 = T.params_to_override(off)
    assert ov0["loss"]["eta"] == 0.0 and ov0["loss"]["den_t_range"] is None and "budget" not in ov0
    r = resolve_cfg("v2_next", deep_merge(cfg, {"methods": {"v2_next": ov}}))
    assert r["loss"]["alpha"] == 1.0 and r["loss"]["den_t_range"] == [300, 700]
    assert r["purifier_train_probs"]["sd_onestep"] == 0.15 and abs(sum(r["purifier_train_probs"].values()) - 1) < 1e-12
    r0 = resolve_cfg("v2_next", deep_merge(cfg, {"methods": {"v2_next": ov0}}))
    assert r0["loss"]["den_t_range"] is None and r0["purifier_train_probs"]["sd_onestep"] == 0.0
    # old-space dicts still use the old mapping (no p_onestep / den_t_range)
    assert "p_onestep" not in T.params_to_override({"alpha": 1, "eta": 2, "beta": 0.5, "gamma": 30, "antipure": False, "ap_w": 0.5})


def test_v2next_search_enqueues_anchor_and_respects_space(env):
    T.main(v2n_args(env, 5))
    study = T.make_study(T.argparse.Namespace(storage=f"sqlite:///{env/'t.db'}", study="s", seed=1, startup=2))
    tr = study.get_trials()
    assert len(tr) == 5 and all(t.state.name == "COMPLETE" for t in tr)
    a = tr[0].params                                              # trial 0 = anchor = current final
    assert a == {"alpha": 0.55, "beta": 0.9023875567667267, "gamma": 60.52658429696642, "den_on": False, "p_onestep": 0.0}
    for t in tr:
        q = t.params
        assert 0.1 <= q["alpha"] <= 2.0 and 0.0 <= q["beta"] <= 1.5 and 20 <= q["gamma"] <= 150
        assert q["p_onestep"] in (0.0, 0.15, 0.3) and "antipure" not in q
        if q["den_on"]:
            assert 0.01 <= q["eta"] <= 1.0 and q["den_t"] in ("50-300", "300-700")
        else:
            assert "eta" not in q and "den_t" not in q
        assert len(t.user_attrs["constraints"]) == 2 and len(t.user_attrs["psnr_each"]) == 2
        # per_photo constraint = [38 - min psnr, 0.95 - min ssim]
        assert t.user_attrs["constraints"] == pytest.approx([38 - min(t.user_attrs["psnr_each"]),
                                                              0.95 - min(t.user_attrs["ssim_each"])])


def test_v2next_confirm_per_photo_and_decision_accept(env, monkeypatch):
    import yaml

    monkeypatch.setattr(T, "PSNR_MIN", 0.0)            # make every photo pass so the stub (psr 0 == anchor 0) is accepted
    monkeypatch.setattr(T, "SSIM_MIN", 0.0)
    (env / "conf.txt").write_text("v0.png\nv1.png\nv2.png")
    T.main(v2n_args(env, 4))
    T.main(v2n_args(env, 4, ["--confirm-top", "2", "--confirm-list", str(env / "conf.txt"), "--confirm-iters", "2",
                             "--confirm-k", "1"]))
    d = env / "out" / "s"
    res = json.load(open(d / "confirm.json"))
    assert "hand_set" in res or "trial_0" in res
    for v in res.values():
        assert set(v["per_photo"]) == {"v0.png", "v1.png", "v2.png"}        # --confirm-list, all of it (confirm-images 0)
        assert all(set(x) == {"psnr", "ssim"} for x in v["per_photo"].values()) and v["n_violations"] == 0
    dec = json.load(open(d / "decision.json"))
    assert dec["accepted"] is True and dec["winner"] != dec["anchor_key"] and dec["winner_psr"] >= dec["anchor_psr"]
    fc = yaml.safe_load(open(d / "final_candidate.yaml"))
    assert list(fc["methods"]) == ["v2_next_tuned"] and "p_onestep" in fc["methods"]["v2_next_tuned"]
    from src.protect.v2_registry import deep_merge, resolve_cfg

    r = resolve_cfg("v2_next_tuned", deep_merge(T.load_cfg(str(env / "config" / "config.yaml")), fc))
    assert r["loss"]["gamma"] == fc["methods"]["v2_next_tuned"]["loss"]["gamma"]


def test_v2next_confirm_rejects_when_photos_violate(env):
    T.main(v2n_args(env, 4))
    # real thresholds: 64 px noise images with 2 PGD iterations at eps 4 cannot all reach 38 dB / 0.95 SSIM
    T.main(v2n_args(env, 4, ["--confirm-top", "2", "--confirm-images", "2", "--confirm-iters", "2", "--confirm-k", "1"]))
    d = env / "out" / "s"
    res = json.load(open(d / "confirm.json"))
    dec = json.load(open(d / "decision.json"))
    viol = {k: v["n_violations"] for k, v in res.items() if k != dec["anchor_key"]}
    assert dec["accepted"] == (any(n == 0 for n in viol.values()))   # consistent with the rule applied to what was measured
    if not dec["accepted"]:
        assert not (d / "final_candidate.yaml").exists()


def _r(psr, viol):
    return {"psr": psr, "n_violations": viol, "params": {}}


def test_decision_rule_picks_and_rejects():
    ok = T.decide({"hand_set": _r(0.30, 3), "trial_4": _r(0.40, 0), "trial_7": _r(0.35, 0)}, "hand_set")
    assert ok["accepted"] and ok["winner"] == "trial_4"                       # best violation-free candidate
    tie = T.decide({"hand_set": _r(0.30, 5), "trial_4": _r(0.30, 0)}, "hand_set")
    assert tie["accepted"]                                                    # >= : a tie with the anchor is accepted
    low = T.decide({"hand_set": _r(0.30, 0), "trial_4": _r(0.29, 0)}, "hand_set")
    assert not low["accepted"] and low["winner"] == "trial_4" and "did not beat" in low["reason"]
    viol = T.decide({"hand_set": _r(0.10, 0), "trial_4": _r(0.90, 1), "trial_7": _r(0.80, 2)}, "hand_set")
    assert not viol["accepted"] and viol["winner"] is None                    # higher PSR but any violation -> rejected
    skip = T.decide({"hand_set": _r(0.30, 0), "trial_4": _r(0.90, 1), "trial_7": _r(0.31, 0)}, "hand_set")
    assert skip["accepted"] and skip["winner"] == "trial_7"                   # violating trial is skipped, not chosen
    only = T.decide({"trial_0": _r(0.5, 0)}, "trial_0")
    assert not only["accepted"]                                               # anchor alone: nothing to promote
