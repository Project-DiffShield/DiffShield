"""Bootstrap CI, PSR thresholds, paired tests aligned by image_id, gate logic (all synthetic, CPU)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.eval import analysis as A
from src.eval.analysis import Spec
from src.eval.stats import bootstrap_ci


def test_boot_ci_matches_stats_module_and_is_deterministic():
    x = np.random.default_rng(1).normal(size=57)
    a = A.boot_ci(x, 2000, 0.95, 2026)
    b = bootstrap_ci(x, n_boot=2000, ci=0.95, seed=2026)
    assert a == pytest.approx(b, abs=1e-12)
    assert A.boot_ci(x, 2000, 0.95, 2026) == a


def test_boot_ci_known_bernoulli_width():
    # 200 successes of 400: analytic 95% half-width = 1.96*sqrt(.25/400) = 0.049
    x = np.r_[np.ones(200), np.zeros(200)]
    m, lo, hi = A.boot_ci(x, 4000, 0.95, 7)
    assert m == 0.5
    assert lo < 0.5 < hi
    assert (hi - lo) / 2 == pytest.approx(0.049, rel=0.12)


def test_boot_ci_coverage_on_normal_data():
    rng = np.random.default_rng(3)
    hit = 0
    for _ in range(150):
        _, lo, hi = A.boot_ci(rng.normal(0, 1, 40), 300, 0.95, 5)
        hit += lo <= 0 <= hi
    assert 0.88 <= hit / 150 <= 0.995


def test_boot_ci_constant_and_empty():
    assert A.boot_ci(np.full(10, 0.3), 100) == (pytest.approx(0.3), pytest.approx(0.3), pytest.approx(0.3))
    assert all(math.isnan(v) for v in A.boot_ci([], 100))
    m, lo, hi = A.boot_ci([1.0, np.nan, 3.0], 100)
    assert m == 2.0          # non-finite values ignored


def test_wilcoxon_all_zero_and_signal():
    assert A.wilcoxon_p(np.zeros(10)) == (1.0, 0)
    p, n = A.wilcoxon_p(np.arange(1, 21, dtype=float))
    assert n == 20 and p < 1e-3


def test_paired_alignment_by_image_id_not_position():
    ids = [f"i{k}" for k in range(30)]
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(1.0, 0.2, 30), index=ids)
    b = pd.Series(rng.normal(0.0, 0.2, 30), index=ids)
    ref = A.paired_stats(a, b, 500)
    shuffled = b.sample(frac=1.0, random_state=1)                       # different row order, same labels
    extra = pd.concat([shuffled, pd.Series([99.0], index=["only_in_b"])])
    got = A.paired_stats(a, extra, 500)
    assert got["n_pairs"] == 30
    assert got["mean_diff"] == pytest.approx(ref["mean_diff"])
    assert got["ci_lo"] == pytest.approx(ref["ci_lo"])
    # a naive positional subtraction would differ
    naive = a.to_numpy() - extra.iloc[:30].to_numpy()
    assert abs(float(np.var(naive)) - float(np.var((a - b).to_numpy()))) > 1e-6


def test_paired_ratio():
    a = pd.Series([0.2, 0.4, 0.6, 0.8], index=list("abcd"))
    b = pd.Series([0.1, 0.2, 0.3, 0.4], index=list("dcba")[::-1])
    r = A.paired_ratio(a, b, 300)
    assert r["ratio"] == pytest.approx(2.0) and r["n_pairs"] == 4
    assert r["ci_lo"] == pytest.approx(2.0) and r["ci_hi"] == pytest.approx(2.0)


# ---------------------------------------------------------------- thresholds
def _unit(v):
    return v / np.linalg.norm(v)


def test_threshold_is_95th_percentile_of_impostor_sims():
    rng = np.random.default_rng(0)
    ids = [f"s{i}" for i in range(12)]
    src = {i: _unit(rng.normal(size=16)).astype(np.float32) for i in ids}
    edits = {}
    for ed in ("sd15_img2img", "ip2p"):
        for i in ids:
            edits[f"{ed}/p0_s0.55_seed1/{i}"] = _unit(src[i] + 0.5 * rng.normal(size=16)).astype(np.float32)
    thr = A.thresholds_from_embeddings(src, edits, 95.0)
    sims = []
    for i in ids:
        for j in ids:
            if i != j:
                sims.append(float(np.dot(edits[f"sd15_img2img/p0_s0.55_seed1/{i}"], src[j])))
    assert thr["per_editor"]["sd15_img2img"] == pytest.approx(np.percentile(sims, 95), abs=1e-6)
    assert thr["n_pairs"]["sd15_img2img"] == 12 * 11
    assert "pooled" in thr and thr["n_pairs"]["pooled"] == 2 * 12 * 11
    # restricting ids shrinks the impostor set
    sub = A.thresholds_from_embeddings(src, edits, 95.0, restrict_ids=set(ids[:6]))
    assert sub["n_pairs"]["sd15_img2img"] == 6 * 5
    # an edit of person i is never compared with source i (own identity excluded)
    assert A.threshold_for(thr, "unknown_editor", "per_editor") == thr["pooled"]
    assert A.threshold_for(None, "x") != A.threshold_for(None, "x")  # nan


def _edit_rows(method, vals, nf=(), purifier="none", editor="sd15_img2img", strength="0.55", split="test"):
    rows = []
    for i, v in enumerate(vals):
        for s in (1, 2, 3):
            rows.append({"run_id": "T", "method": method, "image_id": f"im{i}", "split": split, "seed": s, "editor": editor,
                         "prompt_id": "p0", "strength": strength, "purifier": purifier,
                         "face_detected": 0 if i in nf else 1, "arcface_sim": np.nan if i in nf else v, "edit_lpips": 0.3,
                         "clip_i": 0.8, "inv_psnr": 40.0, "inv_ssim": 0.97, "inv_lpips": 0.2, "eps_255": 4, "n_faces": 1, "det_score": 0.9})
    return rows


def _analysis(rows, thr=0.2, n_boot=300):
    df = pd.DataFrame(rows)
    return A.Analysis(A.typed_edits(df), {"per_editor": {"sd15_img2img": thr}, "pooled": thr}, n_boot=n_boot)


def test_psr_hit_rules_no_face_is_success_and_threshold_strict():
    # sims: below, equal to thr (not a hit: strict <), above; image 3 has no face (success)
    an = _analysis(_edit_rows("m", [0.1, 0.2, 0.5, 0.9], nf=(3,)))
    g = an.frame(Spec("m"))
    assert g.loc["im0", "psr"] == 1.0 and g.loc["im1", "psr"] == 0.0 and g.loc["im2", "psr"] == 0.0
    assert g.loc["im3", "psr"] == 1.0 and g.loc["im3", "no_face"] == 1.0
    st = an.cell(Spec("m"))
    assert st["psr"]["mean"] == 0.5 and st["n_images"] == 4 and st["seeds_max"] == 3 and st["no_face"]["mean"] == 0.25


def test_missing_threshold_gives_nan_psr_not_a_number():
    df = pd.DataFrame(_edit_rows("m", [0.1, 0.5]))
    an = A.Analysis(A.typed_edits(df), {}, n_boot=100)
    assert math.isnan(an.cell(Spec("m"))["psr"]["mean"])
    assert A.cell_text(an.cell(Spec("m")), "psr") == "n/a"
    assert A.cell_text(None) == A.NOT_RUN


def test_analysis_pairs_by_image_id_across_shuffled_rows():
    base = _edit_rows("a", [0.1, 0.1, 0.5, 0.5]) + _edit_rows("b", [0.5, 0.5, 0.5, 0.1])
    rng = np.random.default_rng(0)
    rng.shuffle(base)
    an = _analysis(base)
    pr = an.paired(Spec("a"), Spec("b"), "psr")
    assert pr["n_pairs"] == 4
    # a hits: im0, im1; b hits: im3 -> diff = [1, 1, 0, -1] -> mean 0.25
    assert pr["mean_diff"] == pytest.approx(0.25)
    assert an.paired(Spec("a"), Spec("zzz")) is None


def test_default_strength_resolution():
    rows = _edit_rows("m", [0.1, 0.3], editor="ip2p", strength="")
    an = _analysis(rows)
    assert an.cell(Spec("m", "none", "ip2p"))["n_images"] == 2
    assert an.cell(Spec("m", "none", "ip2p", "0.55")) is None


# ---------------------------------------------------------------- gates
def _gate_analysis(psr_by_cell, n=20):
    """psr_by_cell {(method, purifier): fraction of images that are hits}."""
    rows = []
    for (m, p), frac in psr_by_cell.items():
        k = int(round(frac * n))
        vals = [0.05] * k + [0.9] * (n - k)
        rows += _edit_rows(m, vals, purifier=p)
    return _analysis(rows)


def test_gate_e4_pass_fail_and_degenerate():
    gp = A.GateParams()
    pur = gp.e4_purifiers
    ok = {("v1", p): 0.2 for p in pur} | {("v2_cheap", p): 0.5 for p in pur}
    assert A.gate_e4(_gate_analysis(ok), gp).verdict == "PASS"
    bad = {("v1", p): 0.2 for p in pur} | {("v2_cheap", p): (0.3 if p != pur[0] else 0.5) for p in pur}
    assert A.gate_e4(_gate_analysis(bad), gp).verdict == "FAIL"
    degenerate = {("v1", p): 0.6 for p in pur} | {("v2_cheap", p): 0.9 for p in pur}
    assert A.gate_e4(_gate_analysis(degenerate), gp).verdict.startswith("DEGENERATE")
    assert A.gate_e4(_gate_analysis({("v1", "none"): 0.5}), gp).verdict == "NOT RUN"


def test_gate_e5_half_of_no_purifier_psr():
    gp = A.GateParams()
    ok = {("v2_main", "none"): 0.8, ("v2_main", "diffpure_0.1"): 0.4}
    assert A.gate_e5(_gate_analysis(ok), gp).verdict.startswith("PASS")
    bad = {("v2_main", "none"): 0.8, ("v2_main", "diffpure_0.1"): 0.3}
    assert A.gate_e5(_gate_analysis(bad), gp).verdict == "FAIL"
    assert A.gate_e5(_gate_analysis({("v2_main", "none"): 0.8}), gp).verdict == "NOT RUN"


def test_gate_e3_collapse_vs_barely():
    gp = A.GateParams()
    core = gp.e3_core
    collapse = {("v1", "none"): 0.6} | {("v1", p): 0.05 for p in core}
    assert A.gate_e3(_gate_analysis(collapse), gp).verdict.startswith("DEGRADES")
    barely = {("v1", "none"): 0.6} | {("v1", p): 0.55 for p in core}
    assert A.gate_e3(_gate_analysis(barely), gp).verdict.startswith("BARELY")
    assert A.gate_e3(_gate_analysis({("v1", "none"): 0.6, ("v1", "jpeg_q75"): 0.1}), gp).verdict in ("PARTIAL", "NOT RUN")


def test_gate_param_override():
    gp = A.parse_gate_params(["e4_factor=3", "e4_purifiers=jpeg_q75,vae_rt", "e5_purifier=diffpure_0.2"])
    assert gp.e4_factor == 3.0 and gp.e4_purifiers == ("jpeg_q75", "vae_rt") and gp.e5_purifier == "diffpure_0.2"


def test_dedup_prefers_lpips_then_latest_and_counts_conflicts():
    base = {c: "" for c in A.EDIT_COLUMNS}
    r1 = dict(base, method="v1", image_id="1", seed="1", editor="e", prompt_id="p0", strength="0.550", purifier="none",
              arcface_sim="0.3", edit_lpips="nan", created_utc="2026-10-15T10:00:00Z")
    r2 = dict(r1, strength="0.55", edit_lpips="0.4", created_utc="2026-10-15T09:00:00Z", arcface_sim="0.31")
    r3 = dict(r1, image_id="2")
    out, st = A.dedup_edits(pd.DataFrame([r1, r2, r3]))
    assert len(out) == 2 and st["duplicates_dropped"] == 1 and st["conflicts"] == 1
    kept = out[out["image_id"] == "1"].iloc[0]
    assert kept["edit_lpips"] == "0.4"          # the row with LPIPS wins even though it is older
