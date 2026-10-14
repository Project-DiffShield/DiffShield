import os

import pytest
import torch
import yaml

from test_protect_fakes import TinySem, base_cfg, make_image, make_models  # noqa: F401

from src.protect.pgd import to_u8
from src.protect.v2_registry import PRESETS, build_v2_protector, deep_merge, resolve_cfg

PART = os.path.join(os.path.dirname(__file__), "..", "config", "parts", "protect.yaml")


def full_cfg(**kw):
    c = deep_merge(base_cfg(iters=2, k=2), yaml.safe_load(open(PART)))
    return deep_merge(c, kw)


def test_part_file_defines_all_presets():
    methods = yaml.safe_load(open(PART))["methods"]
    for n in PRESETS:
        assert n in methods or n == "v2_final"
    assert "v2_final" not in methods  # tuned final.yaml owns it (sorts before protect.yaml)
    assert "v2_final_default" in methods


@pytest.mark.parametrize("name", PRESETS)
def test_every_preset_constructs_and_resolves_without_loading_models(name):
    p = build_v2_protector(name, full_cfg(), "cpu")
    assert not p._built
    assert p.params.eps_255 in (2, 4, 6, 8)
    assert p.cfg["method_name"] == name


def test_preset_contents():
    r = {n: resolve_cfg(n, full_cfg()) for n in PRESETS}
    assert r["v2_cheap"]["purifiers_train"] == ["jpeg", "downup", "vae_roundtrip"] and r["v2_cheap"]["loss"]["eta"] == 0
    assert "sd_onestep" in r["v2_main"]["purifiers_train"] and r["v2_main"]["loss"]["eta"] == 0.2699475469647097
    assert r["v2_antipure"]["loss"]["anti_purify"]["enabled"] is True
    assert r["v2_main"]["loss"]["anti_purify"]["enabled"] is False
    assert r["abl_vis_only"]["loss"]["eta"] == 0 and r["abl_vis_only"]["loss"]["beta"] == 0
    assert r["abl_den_only"]["loss"]["alpha"] == 0 and r["abl_den_only"]["loss"]["eta"] > 0
    assert r["abl_no_sem"]["loss"]["beta"] == 0 and r["abl_no_sem"]["loss"]["alpha"] > 0
    assert r["abl_no_pur"]["purifiers_train"] == []
    assert r["v2_final"]["protect"]["ensemble"]["enabled"] is True
    assert [r[f"eps{e}"]["budget"]["eps_255"] for e in (2, 6, 8)] == [2, 6, 8]
    assert r["eps6"]["protect"]["ensemble"]["enabled"] is True  # eps* are variants of v2_final


def test_tuned_final_layer_overrides_default_and_flows_to_eps_variants():
    tuned = {"methods": {"v2_final": {"loss": {"alpha": 0.5, "beta": 0.0, "anti_purify": {"enabled": True}}}}}
    c = full_cfg(**tuned)
    f = resolve_cfg("v2_final", c)
    assert f["loss"]["alpha"] == 0.5 and f["loss"]["beta"] == 0.0 and f["loss"]["anti_purify"]["enabled"]
    assert f["loss"]["gamma"] == 60.52658429696642  # untouched values kept from the default
    e = resolve_cfg("eps8", c)
    assert e["loss"]["alpha"] == 0.5 and e["budget"]["eps_255"] == 8
    # a non-final preset ignores the tuned layer
    assert resolve_cfg("v2_main", c)["loss"]["alpha"] == 0.2699475469647097


def test_unknown_preset_and_idempotent_resolution():
    with pytest.raises(KeyError):
        build_v2_protector("v3_nope", full_cfg(), "cpu")
    once = resolve_cfg("v2_antipure", full_cfg())
    twice = resolve_cfg("v2_antipure", once)
    assert once["loss"] == twice["loss"] and once["purifiers_train"] == twice["purifiers_train"]


def test_registry_works_with_minimal_cfg_via_part_file():
    p = build_v2_protector("v2_main", {}, "cpu")
    assert p.params.alpha == 0.2699475469647097 and "sd_onestep" in p.cfg["purifiers_train"]


@pytest.mark.parametrize("name", PRESETS)
def test_protect_runs_for_every_preset_with_injected_components(name):
    m = make_models()
    comps = {"models": m, "sem": TinySem(), "extra_vaes": {"sdxl_vae_fp16": make_models(seed=4).vae}}
    cfg = full_cfg()
    cfg["protect"] = {"quantize": True}
    p = build_v2_protector(name, cfg, "cpu", comps)
    x = make_image()
    xa, info = p.protect(x, seed=5)
    eps = p.params.eps_255
    assert xa.shape == x.shape and info["method"] == name
    assert (to_u8(xa) - to_u8(x)).abs().max() <= eps
    xb, _ = build_v2_protector(name, cfg, "cpu", comps).protect(x, seed=5)
    assert torch.equal(xa, xb)  # determinism by seed through the public API


def test_ablation_term_selection_reaches_the_loop():
    m = make_models()
    comps = {"models": m, "sem": TinySem()}
    for name, expect in {"abl_vis_only": {"vis", "str"}, "abl_den_only": {"den", "str"},
                         "abl_no_sem": {"vis", "den", "str"}, "v2_cheap": {"vis", "sem", "str"},
                         "v2_main": {"vis", "den", "sem", "str"}}.items():
        _, info = build_v2_protector(name, full_cfg(), "cpu", comps).protect(make_image(), 0)
        assert set(info["terms_init"]) == expect, name
    _, info = build_v2_protector("v2_antipure", full_cfg(loss={"anti_purify": {"enabled": True}}), "cpu",
                                 comps).protect(make_image(), 0)
    assert info["purifier_counts"]
    _, info = build_v2_protector("abl_no_pur", full_cfg(), "cpu", comps).protect(make_image(), 0)
    assert list(info["purifier_counts"]) == ["none"]
