"""Human study: form generation (offline, no key inside) and analysis on synthetic rater answers."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.eval import analysis as A

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mk = _load("make_human_study")
an = _load("analyze_human_study")


def _img(path, v, size=24):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(v)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    t = tmp_path_factory.mktemp("hs")
    ids = [f"{70001 + 13 * i}" for i in range(14)]
    for k, iid in enumerate(ids):
        _img(t / "src" / f"{iid}.png", k)
        _img(t / "prot" / f"{iid}.png", 100 + k)
        _img(t / "job" / "edits" / A.edit_rel_path("v2_final", "none", "sd15_img2img", "p0", "0.55", 1, iid).__add__(".jpg"), 200 + k)
        _img(t / "clean_edits" / (A.clean_rel_path("sd15_img2img", "p0", "0.55", 1, iid) + ".jpg"), 300 + k)
    out = t / "out"
    rc = mk.main(["--source_dir", str(t / "src"), "--protected_dir", str(t / "prot"), "--edits_root", str(t / "job"),
                  "--clean_root", str(t / "clean_edits"), "--n_a", "6", "--n_b", "6", "--out", str(out)])
    assert rc == 0
    return out, ids


def test_forms_are_offline_and_contain_no_key(study):
    out, ids = study
    for v in (1, 2):
        html = (out / "for_raters" / f"study_v{v}.html").read_text(encoding="utf-8")
        assert not re.search(r"(src|href)\s*=\s*['\"]https?://", html)              # no external resources
        assert "fetch(" not in html and "XMLHttpRequest" not in html and "sendBeacon" not in html
        assert "Blob" in html and "download" in html                                # saves a CSV locally
        text = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "DATA", html)   # base64 payloads contain digits
        for iid in ids:                                                              # no image ids / condition leak
            assert iid not in text
        for word in ("v2_final", "protected_side", "clean_edit", "protected_edit"):
            assert word not in text
        assert html.count("data:image/png;base64,") == 12                            # Task A: 6 pairs, lossless
        assert html.count("data:image/jpeg;base64,") == 12                           # Task B: 6 items x 2
    key = pd.read_csv(out / "private" / "answer_key.csv", dtype=str)
    assert set(key["task"]) == {"A", "B"} and not (out / "for_raters" / "answer_key.csv").exists()
    assert (out / "FACILITATOR_README.txt").is_file() and (out / "answers").is_dir()


def test_key_is_balanced_and_versions_swap_conditions(study):
    out, _ = study
    key = pd.read_csv(out / "private" / "answer_key.csv", dtype=str, keep_default_na=False)
    a1 = key[(key["version"] == "1") & (key["task"] == "A")]
    assert sorted(a1["protected_side"].value_counts().tolist()) == [3, 3]            # sides balanced
    b1 = key[(key["version"] == "1") & (key["task"] == "B")].set_index("image_id")["condition"]
    b2 = key[(key["version"] == "2") & (key["task"] == "B")].set_index("image_id")["condition"]
    assert (b1.value_counts() == 3).all()
    assert (b1.loc[b2.index] != b2).all()                                            # every photo flips condition in version 2
    a2 = key[(key["version"] == "2") & (key["task"] == "A")]
    assert set(a1["image_id"]) == set(a2["image_id"])


def _answers(study_out, n_raters, p_correct, p_same_clean, p_same_prot, seed=0, version_alternate=True, drop_after=None):
    rng = np.random.default_rng(seed)
    key = pd.read_csv(study_out / "private" / "answer_key.csv", dtype=str, keep_default_na=False)
    (study_out / "answers").mkdir(exist_ok=True)
    for r in range(n_raters):
        v = str(1 + (r % 2 if version_alternate else 0))
        rows = []
        for k in key[key["version"] == v].itertuples(index=False):
            if k.task == "A":
                other = "L" if k.protected_side == "R" else "R"
                resp = k.protected_side if rng.random() < p_correct else other
            else:
                ps = p_same_clean if k.condition == "clean_edit" else p_same_prot
                resp = "same" if rng.random() < ps else "different"
            rows.append({"study_version": v, "rater_id": f"R{r:02d}", "task": k.task, "item_id": k.item_id, "response": resp,
                         "order_index": 1, "response_ms": 1000, "timestamp_utc": "x"})
        rows.append({"study_version": v, "rater_id": f"R{r:02d}", "task": "meta", "item_id": "device", "response": "laptop",
                     "order_index": 0, "response_ms": 0, "timestamp_utc": "x"})
        pd.DataFrame(rows).to_csv(study_out / "answers" / f"diffshield_study_R{r:02d}_v{v}.csv", index=False)


def test_analysis_chance_level_pilot_flag_and_ci(study, tmp_path):
    out, _ = study
    for f in (out / "answers").glob("*.csv"):
        f.unlink()
    _answers(out, 4, 0.5, 0.8, 0.2, seed=1)                      # only 4 raters -> pilot
    ans, warns = an.load_answers(out / "answers")
    key = pd.read_csv(out / "private" / "answer_key.csv", dtype=str, keep_default_na=False)
    s = an.analyze(ans, key, min_raters=10, n_boot=300)
    assert s["pilot"] is True and s["n_raters"] == 4
    a = s["task_a"]
    assert a["n_trials"] == 24 and a["n_raters"] == 4
    assert a["ci_lo"] <= a["accuracy"] <= a["ci_hi"] and 0 <= a["p_vs_chance"] <= 1
    assert s["devices"] == {"laptop": 4}
    md = an.to_markdown(s)
    assert "PILOT" in md and "4" in md


def test_analysis_known_accuracy_and_rates(study):
    out, _ = study
    for f in (out / "answers").glob("*.csv"):
        f.unlink()
    _answers(out, 12, 1.0, 1.0, 0.0, seed=2)                     # perfect raters; 'same' only for clean edits
    ans, _ = an.load_answers(out / "answers")
    key = pd.read_csv(out / "private" / "answer_key.csv", dtype=str, keep_default_na=False)
    s = an.analyze(ans, key, min_raters=10, n_boot=300)
    assert s["pilot"] is False and s["n_raters"] == 12
    a = s["task_a"]
    assert a["accuracy"] == 1.0 and a["p_vs_chance"] < 1e-6 and a["ci_lo"] > 0.9
    b = s["task_b"]
    assert b["clean_edit"]["same_rate"] == 1.0 and b["protected_edit"]["same_rate"] == 0.0
    assert b["difference"]["clean_minus_protected_pooled"] == 1.0 and b["difference"]["p_fisher_pooled"] < 1e-6
    assert b["clean_edit"]["n_trials"] + b["protected_edit"]["n_trials"] == 12 * 6
    assert an.main(["--study_dir", str(out), "--n_boot", "200"]) == 0
    assert (out / "summary.json").is_file() and (out / "summary.md").is_file() and (out / "per_rater.csv").is_file()


def test_analysis_wrong_key_and_incomplete_and_duplicates(study):
    out, _ = study
    for f in (out / "answers").glob("*.csv"):
        f.unlink()
    _answers(out, 2, 0.5, 0.5, 0.5, seed=3)
    # duplicate a file: last answer wins, a warning is produced
    f = next((out / "answers").glob("*.csv"))
    d = pd.read_csv(f, dtype=str)
    d.to_csv(out / "answers" / "dup_copy.csv", index=False)
    ans, warns = an.load_answers(out / "answers")
    assert any("duplicate" in w for w in warns)
    key = pd.read_csv(out / "private" / "answer_key.csv", dtype=str, keep_default_na=False)
    bad_key = key.assign(item_id="Z" + key["item_id"])
    s = an.analyze(ans, bad_key, n_boot=100)
    assert any("did not match" in w for w in s["warnings"]) and s["n_raters"] == 0
    # incomplete rater excluded with min_items
    ans2 = ans[~((ans["rater_id"] == "R00") & (ans["task"] == "A") & (ans["item_id"] != "A01"))]
    s2 = an.analyze(ans2, key, min_items=3, n_boot=100)
    assert "R00" in s2["raters_incomplete_a"]
