"""make_tables / make_figures end to end on synthetic CSVs; legacy Phase 0 table; template rendering."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.eval import analysis as A
from src.eval.schema import PROTECT_COLUMNS

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mt = _load("make_tables")
mf = _load("make_figures")
ing = _load("ingest_results")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn_tables")
    A.write_synthetic_tree(root, n_images=12)
    ing.ingest(root, root / "combined", log=lambda *a: None)
    out = root / "tables"
    rc = mt.main(["--combined", str(root / "combined"), "--out", str(out), "--n_boot", "200"])
    assert rc == 0
    return root, out


EXPECTED = ["coverage", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e7_editors", "e7_paired", "e8", "e9", "e10", "headline",
            "headline_full", "per_editor", "per_prompt", "messy", "invisibility_methods", "paired_vs_v1", "paired_vs_noise",
            "claims", "gates"]


def test_every_table_written_in_four_formats(built):
    _, out = built
    for n in EXPECTED:
        for ext in ("md", "csv", "tex"):
            assert (out / f"{n}.{ext}").is_file(), (n, ext)
    assert (out / "headline_long.csv").is_file() and (out / "numbers.json").is_file() and (out / "gates.json").is_file()


def test_headline_content_not_run_banner_and_n(built):
    _, out = built
    md = (out / "headline.md").read_text()
    assert "SYNTHETIC DATA" in md                                    # banner: never mistaken for results
    assert "not run" in md                                           # e.g. noise controls were not purified
    assert "n images per cell: 12" in md and "edit seeds per image: 3" in md
    assert "95% percentile-bootstrap CI" in md
    df = pd.read_csv(out / "headline.csv")
    assert df.loc[df["method"].str.startswith("v1"), "none"].iloc[0].count("[") == 1
    long = pd.read_csv(out / "headline_long.csv")
    assert {"n_images", "seeds_max", "psr", "psr_lo", "psr_hi"} <= set(long.columns)
    tex = (out / "headline.tex").read_text()
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex and "jpeg\\_q75" in tex
    assert "SYNTHETIC DATA" in tex


def test_numbers_and_gates_json(built):
    _, out = built
    nums = json.loads((out / "numbers.json").read_text())
    assert nums["synthetic"].startswith("YES")
    assert "psr.v1.none" in nums and "gate.E4" in nums and "inv.v1.psnr" in nums and "claim.B" in nums
    g = json.loads((out / "gates.json").read_text())
    assert set(g) == {"E3", "E4", "E5", "E6", "INV"}
    assert all("verdict" in v and "rule" in v for v in g.values())


def test_e9_maps_epsilon_variants(built):
    _, out = built
    e9 = pd.read_csv(out / "e9.csv")
    assert list(e9["eps (x/255)"]) == [2, 4, 6, 8]
    assert list(e9["method"]) == ["eps2", "v2_final", "eps6", "eps8"]


def test_missing_data_is_not_run_not_fabricated(tmp_path):
    # only E0 clean rows: every other table must say "not run"/n/a instead of producing numbers
    e, p, src, emb = A.synthetic_results(8)
    e = e[(e["method"] == "clean") & (e["purifier"] == "none")]
    comb = tmp_path / "combined"
    comb.mkdir()
    e.to_csv(comb / "edits.csv", index=False)
    (comb / "psr_thresholds.json").write_text(json.dumps(A.thresholds_from_embeddings(src, emb)))
    out = tmp_path / "t"
    assert mt.main(["--combined", str(comb), "--out", str(out), "--n_boot", "100", "--table", "headline,e4,e5,e9,gates"]) == 0
    assert "not run" in (out / "headline.md").read_text()
    gates = json.loads((out / "gates.json").read_text())
    assert gates["E4"]["verdict"] == "NOT RUN" and gates["E5"]["verdict"] == "NOT RUN" and gates["E3"]["verdict"] == "NOT RUN"
    assert "not run" in (out / "e9.md").read_text()


def test_no_thresholds_gives_na_psr(tmp_path):
    e, p, src, emb = A.synthetic_results(6)
    e = e[e["purifier"] == "none"].head(200)
    comb = tmp_path / "combined"
    comb.mkdir()
    e.to_csv(comb / "edits.csv", index=False)
    out = tmp_path / "t"
    assert mt.main(["--combined", str(comb), "--out", str(out), "--n_boot", "100", "--table", "headline"]) == 0
    md = (out / "headline.md").read_text()
    assert "n/a" in md and "NOT available" in md


def test_legacy_phase0_invisibility_table_still_works(tmp_path):
    rd = tmp_path / "res" / "runX"
    rd.mkdir(parents=True)
    rows = []
    for i in range(5):
        r = dict.fromkeys(PROTECT_COLUMNS, 0.0)
        r.update(run_id="runX", method="v1", image_id=str(i), v1_psnr=41.9 + i * 0.01, v1_ssim=0.975, v1_lpips=0.23,
                 v1_linf_model=0.0313725, png_psnr=41.8 + i * 0.01, png_ssim=0.974, png_lpips=0.23, png_maxabs_u8=4)
        rows.append(r)
    pd.DataFrame(rows).to_csv(rd / "metrics.csv", index=False)
    assert mt.main(["--run_id", "runX", "--output_root", str(tmp_path / "res"), "--table", "invisibility"]) == 0
    md = (rd / "tables" / "invisibility.md").read_text()
    assert "PSNR (dB)" in md and "reproduced, v1 convention" in md
    assert (rd / "tables" / "invisibility.csv").is_file()


def test_render_template_fills_tables_numbers_and_marks_pending(built, tmp_path):
    root, out = built
    tpl = tmp_path / "PHASE_X.md"
    tpl.write_text("# T\n{{table:headline}}\nE4 gate: {{gate:E4}}\nv1 PSR {{num:psr_pt.v1.none}}\n"
                   "missing {{num:no.such.key}} and {{table:nope}} and {{num:no.key|n/a}}\n", encoding="utf-8")
    rend = tmp_path / "rendered"
    assert mt.main(["--render", str(tpl), "--tables_dir", str(out), "--out", str(rend)]) == 0
    text = (rend / "PHASE_X.md").read_text()
    assert "| method |" in text and "SYNTHETIC DATA" in text
    assert "PENDING (no.such.key)" in text and "PENDING (table `nope`" in text and "n/a" in text
    assert "{{" not in text


def test_figures_on_synthetic_and_skips_without_images(built, tmp_path):
    root, _ = built
    out = tmp_path / "figs"
    rc = mf.main(["--combined", str(root / "combined"), "--out", str(out), "--n_boot", "100", "--results", str(tmp_path / "none")])
    assert rc == 0
    man = json.loads((out / "figures_manifest.json").read_text())
    for name in ("psr_vs_psnr", "psr_by_purifier", "per_editor", "ablation", "sim_distributions"):
        assert man[name]["files"] == [f"{name}.png", f"{name}.pdf"] and (out / f"{name}.png").stat().st_size > 2000
    assert "skipped" in man["before_after"] and "skipped" in man["optuna_history"] and "skipped" in man["human_study"]
    fc = pd.read_csv(out / "failure_cases.csv")
    assert fc["sim"].is_monotonic_decreasing and fc["method"].eq("v2_final").all()


def test_image_figures_with_tiny_images(built, tmp_path):
    root, _ = built
    comb = root / "combined"
    ed = pd.read_csv(comb / "edits.csv", dtype=str, keep_default_na=False)
    ids = sorted(ed.loc[ed["method"] == "clean", "image_id"].unique())[:3]
    base = tmp_path / "job"
    src = tmp_path / "src"
    for iid in ids:
        src.mkdir(exist_ok=True)
        Image.new("RGB", (32, 32), (90, 90, 90)).save(src / f"{iid}.png")
        for m in ("v1", "v2_final"):
            for pid in ("p0", "p1", "p2"):
                for seed in (1, 2, 3):
                    q = base / "edits" / A.edit_rel_path(m, "none", "sd15_img2img", pid, "0.55", seed, iid)
                    q.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), (120, 60, 60)).save(str(q) + ".jpg")
        for pid in ("p0", "p1", "p2"):
            for seed in (1, 2, 3):
                q = tmp_path / "clean_edits" / (A.clean_rel_path("sd15_img2img", pid, "0.55", seed, iid) + ".jpg")
                q.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (60, 120, 60)).save(q)
    out = tmp_path / "figs2"
    rc = mf.main(["--combined", str(comb), "--out", str(out), "--n_boot", "100", "--fig", "before_after,failure_gallery",
                  "--source_dir", str(src), "--edits_root", str(base), "--clean_root", str(tmp_path / "clean_edits"),
                  "--results", str(tmp_path / "none"), "--n_show", "2"])
    assert rc == 0
    man = json.loads((out / "figures_manifest.json").read_text())
    assert "files" in man["before_after"]
    assert "files" in man["failure_gallery"] or "skipped" in man["failure_gallery"]


def test_optuna_sqlite_reader(tmp_path):
    import sqlite3
    db = tmp_path / "o.db"
    con = sqlite3.connect(db)
    con.executescript("""create table studies(study_id integer, study_name text);
        create table trials(trial_id integer, number integer, study_id integer, state text);
        create table trial_values(trial_value_id integer, trial_id integer, objective integer, value real, value_type text);
        create table trial_user_attributes(trial_user_attribute_id integer, trial_id integer, key text, value_json text);
        insert into studies values (1,'s');
        insert into trials values (1,0,1,'COMPLETE'),(2,1,1,'COMPLETE'),(3,2,1,'FAIL');
        insert into trial_values values (1,1,0,0.2,'FINITE'),(2,2,0,0.5,'FINITE');
        insert into trial_user_attributes values (1,1,'feasible','true'),(2,2,'feasible','false');""")
    con.commit()
    con.close()
    d = mf.read_optuna_sqlite(db)
    assert list(d["number"]) == [0, 1, 2] and d.loc[0, "feasible"] is True or d.loc[0, "feasible"] == True  # noqa: E712
    assert d.loc[1, "feasible"] == False  # noqa: E712


# ---------------------------------------------------------------- templates in reports/
def test_report_templates_reference_only_known_tables_gates_and_resolvable_numbers(built, tmp_path):
    root, out = built
    nums = json.loads((out / "numbers.json").read_text())
    known_external = {"phase0.summary", "optuna.trials", "failures.summary", "human.task_a", "human.pilot", "human.n_raters"}
    files = sorted((REPO / "reports").glob("PHASE_[1-8].md"))
    assert len(files) == 8
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in A._PH.finditer(text):
            kind, key, default = m.group(1), m.group(2).strip(), m.group(3)
            if kind == "table":
                assert key in mt.TABLES, (f.name, key)
            elif kind == "gate":
                assert key in ("E3", "E4", "E5", "E6", "INV"), (f.name, key)
                assert f"gate.{key}" in nums
            else:
                assert key in nums or key in known_external or default, (f.name, key)
        rend, missing = A.render_template(text, out, nums)
        assert "{{" not in rend
    # every table a template includes was actually generated
    d = REPO / "docs" / "REPORT_DRAFT.md"
    if d.exists():
        for m in A._PH.finditer(d.read_text(encoding="utf-8")):
            if m.group(1) == "table":
                assert (out / f"{m.group(2).strip()}.md").is_file()
