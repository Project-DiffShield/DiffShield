"""Tier expansion, prompt rotation, schema constants. CPU only, no models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.eval import edit_runner as er

CFG = er.load_eval_config()
EV = CFG["eval"]
IDS100 = [f"img{i:03d}" for i in range(100)]


def rows(jobs):
    return er.count_rows(jobs, len(EV["seeds"]))


def test_config_loads_and_has_all_tiers():
    for t in ("H", "M", "P", "A", "E9", "E8", "SMOKE"):
        assert t in EV["tiers"]
    assert EV["seeds"] == [1, 2, 3]
    assert EV["num_inference_steps"] == 30
    for ed in EV["editors"].values():
        assert ed["prompts"] in EV["prompts"]
        assert ed["model"] in EV["models"]


def test_tier_h_v1_counts():
    jobs = er.expand_tier(EV, "H", "v1", "test", IDS100)
    c = rows(jobs)
    # sd15: s0.3 (1 prompt) + s0.55 (3 prompts) + s0.75 (1 prompt) = 5 jobs/image x 3 seeds
    assert c["sd15_img2img"] == 100 * 5 * 3
    assert c["ip2p"] == c["sd_inpaint"] == c["ipadapter_faceid"] == 100 * 3
    assert c["sdxl_img2img"] == 100 * 3          # v1 is in the SDXL subset
    assert c["total"] == 1500 + 300 * 4


def test_tier_h_sdxl_only_for_subset():
    assert "sdxl_img2img" not in rows(er.expand_tier(EV, "H", "v2_main", "test", IDS100))
    assert "sdxl_img2img" in rows(er.expand_tier(EV, "H", "v2_final", "test", IDS100))
    assert "sdxl_img2img" in rows(er.expand_tier(EV, "H", "clean", "test", IDS100))


def test_prompt_rotation_and_headline_all_prompts():
    jobs = er.expand_tier(EV, "H", "v1", "test", IDS100, ["none"], ["sd15_img2img"])
    by = {}
    for j in jobs:
        by.setdefault((j.image_id, j.strength), set()).add(j.prompt_idx)
    assert by[("img000", "0.55")] == {0, 1, 2}
    assert by[("img001", "0.3")] == {1}          # rotation: image_index mod 3
    assert by[("img005", "0.75")] == {2}
    assert by[("img003", "0.3")] == {0}
    # non-headline cell (ip2p) is always rotated
    j2 = er.expand_tier(EV, "H", "v1", "test", IDS100, ["none"], ["ip2p"])
    assert all(j.prompt_idx == j.image_index % 3 for j in j2)


def test_tier_p_v1_all_purifiers():
    jobs = er.expand_tier(EV, "P", "v1", "test", IDS100, ["all"])
    c = rows(jobs)
    assert c["sd15_img2img"] == 100 * 13 * 3
    assert c["ipadapter_faceid"] == 100 * 4 * 3
    assert c["ip2p"] == 100 * 2 * 3
    pur = {j.purifier for j in jobs if j.editor == "sd15_img2img"}
    assert len(pur) == 13 and "gridpure" in pur and "none" not in pur


def test_tier_p_photoguard_five_purifiers_and_filter():
    jobs = er.expand_tier(EV, "P", "photoguard", "test", IDS100, ["all"])
    assert {j.purifier for j in jobs} == {"jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt", "diffpure_0.1"}
    only = er.expand_tier(EV, "P", "v1", "test", IDS100, ["jpeg_q75"])
    assert {j.purifier for j in only} == {"jpeg_q75"}
    # alias resolves
    al = er.expand_tier(EV, "P", "v1", "test", IDS100, ["diffpure_t0.1"])
    assert {j.purifier for j in al} == {"diffpure_0.1"}


def test_messy_tier_and_split_gating():
    ids = [f"m{i}" for i in range(20)]
    jobs = er.expand_tier(EV, "M", "v1", "messy", ids)
    c = rows(jobs)
    assert c["sd15_img2img"] == 20 * 3 and c["ipadapter_faceid"] == 20 * 3 and c["total"] == 120
    assert er.expand_tier(EV, "M", "v1", "test", ids) == []        # messy tier does not run on test
    assert er.expand_tier(EV, "M", "abl_no_pur", "messy", ids) == []


def test_ablation_and_eps_patterns():
    a = er.expand_tier(EV, "A", "abl_den_only", "test", IDS100, ["all"])
    assert rows(a)["sd15_img2img"] == 100 * 5 * 3
    e = er.expand_tier(EV, "E9", "eps6", "test", IDS100)
    assert {j.purifier for j in e if j.editor == "sd15_img2img"} == {"none", "jpeg_q75", "diffpure_0.1"}
    assert er.expand_tier(EV, "E9", "v1", "test", IDS100) == []


def test_sharding_partitions_jobs():
    full = er.expand_tier(EV, "H", "clean", "test", IDS100)
    s0 = er.expand_tier(EV, "H", "clean", "test", IDS100, shard=(0, 2))
    s1 = er.expand_tier(EV, "H", "clean", "test", IDS100, shard=(1, 2))
    assert len(s0) + len(s1) == len(full)
    assert not set(s0) & set(s1)
    # prompt id depends on global index, not shard position
    assert {(j.image_id, j.prompt_idx) for j in s1 if j.strength == "0.3"} == \
           {(j.image_id, j.prompt_idx) for j in full if j.strength == "0.3" and j.image_index % 2 == 1}


def test_row_keys_and_strength_format():
    assert er.fmt_strength(0.30) == "0.3" and er.fmt_strength(None) == "" and er.fmt_strength(1.0) == "1"
    j = er.expand_tier(EV, "H", "v1", "test", IDS100[:1], ["none"], ["ip2p"])[0]
    assert j.strength == ""
    keys = j.row_keys([1, 2, 3])
    assert len(set(keys)) == 3 and keys[0][0] == "v1"


def test_schema_columns_match_plan():
    cols = er.ROW_COLUMNS
    for c in ("run_id", "method", "method_cfg_hash", "image_id", "split", "seed", "editor", "prompt_id", "strength",
              "purifier", "protect_run_id", "eps_255", "purifier_params", "editor_params", "source_path",
              "protected_path", "purified_path", "edit_path", "clean_edit_path", "code_git_hash", "created_utc",
              "face_detected", "n_faces", "det_score", "arcface_sim", "fr2_sim", "edit_lpips", "clip_i",
              "inv_psnr", "inv_ssim", "inv_lpips", "gpu_seconds_edit", "gpu_seconds_purify"):
        assert c in cols
    assert len(cols) == len(set(cols))
    assert set(er.KEY_COLUMNS) <= set(cols)


def test_derive_seed_deterministic_and_distinct():
    a = er.derive_seed(2026, "edit", "img1", 1)
    assert a == er.derive_seed(2026, "edit", "img1", 1)
    assert a != er.derive_seed(2026, "edit", "img1", 2) != er.derive_seed(2026, "edit", "img2", 1)
    assert 0 <= a < 2 ** 31
