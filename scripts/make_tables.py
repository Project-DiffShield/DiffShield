"""Generate every table from the results CSVs (never by hand). Each table is written as Markdown, CSV (display),
`<name>_long.csv` (all numbers: n images, seeds, CIs) and LaTeX (booktabs).

Combined mode (after scripts/ingest_results.py):
  python scripts/make_tables.py --combined results/combined --out results/tables [--table all|NAME,NAME]
  tables: coverage e0 e1 e2 e3 e4 e5 e6 e7 e7_editors e7_paired e8 e9 e10 headline headline_full per_editor per_prompt
          messy invisibility_methods paired_vs_v1 paired_vs_noise claims gates     (plus numbers.json)
  --gate_param key=value ...   change a gate operationalisation (see src.eval.analysis.GateParams)
  --n_boot N                   bootstrap resamples (default 10000)
  --synthetic_demo DIR         write a fake Kaggle tree + combined CSVs under DIR and build the tables from it

Render report templates (reports/PHASE_n.md, docs/REPORT_DRAFT.md) with the generated tables/numbers:
  python scripts/make_tables.py --render reports/PHASE_3.md docs/REPORT_DRAFT.md --tables_dir results/tables --out reports/rendered
  Placeholders: {{table:NAME}}  {{num:KEY}}  {{gate:E4}}; anything missing becomes "PENDING (...)".

Phase 0 mode (unchanged): python scripts/make_tables.py --run_id <id> [--output_root results] --table invisibility
Cells without rows print "not run". Every table's note lists n images, seeds and the CI method.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.eval import analysis as A  # noqa: E402
from src.eval.analysis import NOT_RUN, Spec, fmt_ci, fmt_num, fmt_p  # noqa: E402
from src.eval.stats import summarize  # noqa: E402

V1_SUMMARY = REPO / "data" / "v1_ref" / "stage1_summary_metrics.csv"


# ======================================================================================== Phase 0 (legacy)
def _fmt(s, nd=3):
    return f"{s['mean']:.{nd}f} +- {s['std']:.{nd}f} [{s['ci_lo']:.{nd}f}, {s['ci_hi']:.{nd}f}]"


def invisibility(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "metrics.csv")
    ref = {}
    if V1_SUMMARY.exists():
        r = pd.read_csv(V1_SUMMARY).set_index("Metric")
        ref = {"PSNR (dB)": f"{r.loc['PSNR (dB)', 'Mean']:.3f} +- {r.loc['PSNR (dB)', 'Std']:.3f}",
               "SSIM": f"{r.loc['SSIM (Windowed)', 'Mean']:.4f} +- {r.loc['SSIM (Windowed)', 'Std']:.4f}",
               "LPIPS (VGG)": f"{r.loc['LPIPS', 'Mean']:.4f} +- {r.loc['LPIPS', 'Std']:.4f}",
               "Linf [-1,1]": f"{r.loc['Linf', 'Mean']:.5f}"}
    rows = []
    spec = [("PSNR (dB)", "v1_psnr", "png_psnr", lambda x: (x < 38).sum(), "below 38"),
            ("SSIM", "v1_ssim", "png_ssim", lambda x: (x < 0.95).sum(), "below 0.95"),
            ("LPIPS (VGG)", "v1_lpips", "png_lpips", None, "n/a"),
            ("Linf [-1,1]", "v1_linf_model", None, None, "n/a")]
    for name, c1, c2, viol, vtxt in spec:
        s1 = summarize(df[c1].to_numpy(float))
        nd = 5 if name.startswith("Linf") else (4 if name != "PSNR (dB)" else 3)
        row = {"metric": name, "v1 original run (stage1 csv, 70 imgs)": ref.get(name, "n/a"),
               "reproduced, v1 convention": _fmt(s1, nd), "n": s1["n"]}
        if c2:
            s2 = summarize(df[c2].to_numpy(float))
            row["reproduced, published PNG"] = _fmt(s2, nd)
        else:
            row["reproduced, published PNG"] = "n/a"
        row["violations (v1 / png)"] = (f"{int(viol(df[c1]))} / {int(viol(df[c2]))} {vtxt}" if viol else "n/a")
        rows.append(row)
    rows.append({"metric": "PNG max abs gray-level change", "v1 original run (stage1 csv, 70 imgs)": "n/a",
                 "reproduced, v1 convention": "n/a", "n": len(df),
                 "reproduced, published PNG": str(int(df["png_maxabs_u8"].max())), "violations (v1 / png)": "n/a"})
    return pd.DataFrame(rows)


# ===================================================================================== label dictionaries
METHOD_LABEL = {"clean": "clean (E0)", "noise_uniform": "uniform noise (E1)", "noise_gauss_psnr": "Gaussian noise, PSNR-matched (E1)",
                "photoguard": "PhotoGuard encoder attack (E2, reimpl.)", "mist": "Mist-style (E2, reimpl.)",
                "v1": "v1 (E3)", "v2_cheap": "v2 cheap purifiers (E4)", "v2_main": "v2 + one-step SD purifier + L_den (E5)",
                "v2_antipure": "v2 + anti-purification (E6)", "v2_final": "v2 final",
                "abl_vis_only": "L_vis only", "abl_den_only": "L_den only", "abl_no_sem": "both, no L_sem",
                "abl_no_pur": "both, no purifier in loop"}
HEAD_METHODS = ["clean", "noise_uniform", "noise_gauss_psnr", "photoguard", "mist", "v1", "v2_cheap", "v2_main",
                "v2_antipure", "v2_final"]
HEAD_PURS = ["none", "jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt", "diffpure_0.1", "diffpure_0.2"]
EDITOR_CELLS = [("SD1.5 s0.3", A.HEAD_EDITOR, "0.3"), ("SD1.5 s0.55", A.HEAD_EDITOR, "0.55"),
                ("SD1.5 s0.75", A.HEAD_EDITOR, "0.75"), ("SDXL s0.55", "sdxl_img2img", "0.55"),
                ("InstructPix2Pix", "ip2p", ""), ("SD inpaint (keep-face)", "sd_inpaint", "1"),
                ("IP-Adapter FaceID", "ipadapter_faceid", "0.55")]
HELDOUT = ["jpeg_other", "downup_other", "diffpure_0.05", "diffpure_0.1", "diffpure_0.2", "diffpure_0.3",
           "restore_swin2sr", "gridpure"]


def lab(m):
    return METHOD_LABEL.get(m, m)


class Ctx:
    """Everything a table builder needs; records the cells it touches so the note can state n images and seeds."""

    def __init__(self, edits, protect, thresholds, scope="per_editor", n_boot=10000, human=None, gp=None):
        self.edits, self.protect = edits, protect
        self.an = A.Analysis(edits, thresholds, scope, n_boot)
        self.inv = A.invisibility_by_method(protect, edits, n_boot=n_boot)
        self.gp = gp or A.GateParams()
        self.human = human
        self.scope = scope
        self.thresholds = thresholds or {}
        self.n_boot = n_boot
        self.ref_n = self.an.reference_n()
        self.synthetic = bool(len(edits)) and bool(edits["run_id"].astype(str).str.startswith(A.SYNTH_PREFIX).all())
        self._gates = None
        self.optuna_db = None
        self.used: list = []

    @property
    def gates(self):
        if self._gates is None:
            self._gates = A.all_gates(self.an, self.inv, self.gp, self.human)
        return self._gates

    def cell(self, spec):
        st = self.an.cell(spec)
        if st is not None:
            self.used.append(st)
        return st

    def txt(self, spec, metric="psr", nd=2):
        return A.cell_text(self.cell(spec), metric, nd, self.ref_n)

    def reset(self):
        self.used = []

    def note(self, extra: str = "") -> str:
        u = self.used
        parts = []
        if u:
            ns = [s["n_images"] for s in u]
            smin = min(s["seeds_min"] for s in u)
            smax = max(s["seeds_max"] for s in u)
            pmax = max(s["prompts_max"] for s in u)
            parts.append(f"n images per cell: {max(ns)}" + (f" (range {min(ns)}-{max(ns)})" if min(ns) != max(ns) else "")
                         + f"; edit seeds per image: {smin if smin == smax else f'{smin}-{smax}'}; prompts per image: up to {pmax}.")
        else:
            parts.append("no cell of this table has been run yet.")
        parts.append(f"Mean with 95% percentile-bootstrap CI over images ({self.n_boot} resamples, seed 2026); seeds and prompts are "
                     "averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below "
                     f"the {self.scope.replace('_', '-')} threshold (95th percentile of impostor sim of the E0 clean edits)"
                     + (": " + ", ".join(f"{k} {v:.3f}" for k, v in sorted((self.thresholds.get('per_editor') or {}).items()))
                        + (f", pooled {self.thresholds['pooled']:.3f}" if "pooled" in self.thresholds else "") + "." if self.thresholds else
                        ". Thresholds are NOT available yet, so PSR is shown as n/a."))
        parts.append("'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...).")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def long(self, used=None) -> pd.DataFrame:
        rows = []
        for s in (used if used is not None else self.used):
            sp = s["spec"]
            r = {"split": sp.split, "method": sp.method, "purifier": sp.purifier, "editor": sp.editor,
                 "strength": sp.strength, "n_images": s["n_images"], "n_edits": s["n_edits"], "seeds_min": s["seeds_min"],
                 "seeds_max": s["seeds_max"], "prompts_max": s["prompts_max"], "threshold": s["threshold"]}
            for m in A.METRICS + ("inv_psnr", "inv_ssim", "inv_lpips"):
                r[m], r[m + "_lo"], r[m + "_hi"], r[m + "_n"] = s[m]["mean"], s[m]["lo"], s[m]["hi"], s[m]["n"]
            rows.append(r)
        return pd.DataFrame(rows).drop_duplicates(["split", "method", "purifier", "editor", "strength"]) if rows else pd.DataFrame()

    def table(self, name, title, df, extra="") -> A.Table:
        t = A.Table(name, title, df, self.long(), self.note(extra), self.synthetic)
        self.reset()
        return t


# =========================================================================================== table builders
def _matrix(ctx, name, title, methods, cols, spec_fn, metric="psr", extra="", col_label=None):
    """rows = methods, cols = items; spec_fn(method, col) -> Spec."""
    ctx.reset()
    rows = []
    for m in methods:
        r = {"method": lab(m)}
        for c in cols:
            r[col_label(c) if col_label else c] = ctx.txt(spec_fn(m, c), metric)
        rows.append(r)
    return ctx.table(name, title, pd.DataFrame(rows), extra)


def t_headline(ctx):
    return _matrix(ctx, "headline", "PSR by method and purifier, SD1.5 img2img strength 0.55 (test split)", HEAD_METHODS,
                   HEAD_PURS, lambda m, p: Spec(m, p),
                   extra="Non-adaptive mode. Held-out purifiers: jpeg_other, downup_other, diffpure_*, restore_swin2sr, gridpure. "
                         "PhotoGuard and Mist are our reimplementations (not the official code).")


def t_headline_full(ctx):
    return _matrix(ctx, "headline_full", "PSR by method and ALL purifiers, SD1.5 img2img strength 0.55 (test split)",
                   HEAD_METHODS, A.PURIFIER_ORDER[:-1], lambda m, p: Spec(m, p))


def t_per_editor(ctx):
    return _matrix(ctx, "per_editor", "PSR by method and editor, no purifier (test split)", HEAD_METHODS,
                   [c for c in EDITOR_CELLS], lambda m, c: Spec(m, "none", c[1], c[2]), col_label=lambda c: c[0],
                   extra="FaceID reads identity through ArcFace, not the VAE; low PSR there for VAE-based attacks is expected "
                         "(V2_PLAN R10). SDXL ran for clean, v1, PhotoGuard and v2_final only.")


def t_per_prompt(ctx):
    ctx.reset()
    ed = ctx.edits
    sub = ed[(ed["editor"] == A.HEAD_EDITOR) & (ed["strength"] == A.HEAD_STRENGTH) & (ed["purifier"] == "none") & (ed["split"] == "test")]
    rows = []
    prompts = sorted(sub["prompt_id"].unique())
    for m in HEAD_METHODS:
        r = {"method": lab(m)}
        for p in prompts:
            s = sub[(sub["prompt_id"] == p) & (sub["method"] == m)]
            if len(s) == 0:
                r[p] = NOT_RUN
                continue
            an = A.Analysis(s, ctx.thresholds, ctx.scope, ctx.n_boot)
            ctx.used.append(an.cell(Spec(m, "none")))
            r[p] = A.cell_text(an.cell(Spec(m, "none")), "psr", 2, None)
        rows.append(r)
    return ctx.table("per_prompt", "PSR by prompt, SD1.5 s0.55, no purifier (headline cell has all 3 prompts per image)",
                     pd.DataFrame(rows), "p0 'a person in a suit', p1 'a person at a beach', p2 'an oil painting of a person'.")


def t_coverage(ctx):
    cov = A.coverage(ctx.edits)
    d = cov[["split", "method", "purifier", "editor", "strengths", "n_images", "n_rows", "seeds", "n_prompts"]].copy()
    d["face_rate"] = cov["face_rate"].map(lambda v: f"{v:.2f}")
    return A.Table("coverage", "Coverage: cells with rows, images and seeds", d, None,
                   "Rows per (split, method, purifier, editor) cell. See results/combined/coverage_vs_plan.csv for planned-but-missing cells.",
                   ctx.synthetic)


def t_e0(ctx):
    ctx.reset()
    rows = []
    for name, ed, st in EDITOR_CELLS:
        s = ctx.cell(Spec("clean", "none", ed, st))
        if s is None:
            rows.append({"editor": name, "n images": NOT_RUN, "ArcFace sim (clean edit vs source)": NOT_RUN, "PSR of clean edits": NOT_RUN,
                         "no-face rate": NOT_RUN, "CLIP-I": NOT_RUN, "PSR threshold": NOT_RUN})
            continue
        rows.append({"editor": name, "n images": s["n_images"], "ArcFace sim (clean edit vs source)": A.cell_text(s, "sim", 3),
                     "PSR of clean edits": A.cell_text(s, "psr"), "no-face rate": A.cell_text(s, "no_face"),
                     "CLIP-I": A.cell_text(s, "clip_i", 3),
                     "PSR threshold": fmt_num(s["threshold"], 3)})
    return ctx.table("e0", "E0: clean image -> editors (reference edit distribution and PSR threshold)", pd.DataFrame(rows),
                     "The PSR of clean edits is a sanity check: edits of unprotected photos should rarely fall below the impostor threshold.")


def _paired_rows(ctx, comps, cells=None, purs=("none",)):
    """comps = [(method_a, method_b)]; returns rows with a-b differences for each editor cell and purifier."""
    rows = []
    cells = cells or EDITOR_CELLS
    for a, b in comps:
        for pur in purs:
            for name, ed, st in cells:
                sa, sb = Spec(a, pur, ed, st), Spec(b, pur, ed, st)
                ca, cb = ctx.cell(sa), ctx.cell(sb)
                prs, prm = ctx.an.paired(sa, sb, "psr"), ctx.an.paired(sa, sb, "sim")
                if ca is None or cb is None or prs is None:
                    rows.append({"comparison (A vs B)": f"{a} vs {b}", "purifier": pur, "editor": name, "pairs": 0,
                                 "PSR A": NOT_RUN, "PSR B": NOT_RUN, "PSR A-B [95% CI]": NOT_RUN, "p (PSR)": "", "sim A-B [95% CI]": NOT_RUN,
                                 "p (sim)": "", "images A>B": ""})
                    continue
                rows.append({"comparison (A vs B)": f"{a} vs {b}", "purifier": pur, "editor": name, "pairs": prs["n_pairs"],
                             "PSR A": fmt_num(prs["mean_a"]), "PSR B": fmt_num(prs["mean_b"]),
                             "PSR A-B [95% CI]": fmt_ci(prs["mean_diff"], prs["ci_lo"], prs["ci_hi"], 2, True),
                             "p (PSR)": fmt_p(prs["p_wilcoxon"]),
                             "sim A-B [95% CI]": fmt_ci(prm["mean_diff"], prm["ci_lo"], prm["ci_hi"], 3, True) if prm else "n/a",
                             "p (sim)": fmt_p(prm["p_wilcoxon"]) if prm else "n/a",
                             "images A>B": f"{prs['share_a_greater']:.2f}"})
    return pd.DataFrame(rows)


PAIRED_NOTE = ("Paired differences A-B on the same images (aligned by image_id): mean per-image difference with paired image-bootstrap CI and "
               "two-sided Wilcoxon signed-rank p, NOT corrected for the number of comparisons in this table. "
               "PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity (ArcFace sim, 1 = same person).")


def t_e1(ctx):
    ctx.reset()
    df = _paired_rows(ctx, [("noise_uniform", "clean"), ("noise_gauss_psnr", "clean")])
    t = ctx.table("e1", "E1: random-noise controls vs clean images (how much of the effect is just noise?)", df, PAIRED_NOTE)
    return t


def t_e2(ctx):
    ctx.reset()
    df = _paired_rows(ctx, [("photoguard", "noise_uniform"), ("mist", "noise_uniform"), ("v1", "photoguard"), ("v1", "mist")])
    extra = PAIRED_NOTE + " PhotoGuard and Mist are our reimplementations of the encoder/semantic+textural attacks at our budget."
    return ctx.table("e2", "E2: PhotoGuard and Mist vs noise control and v1", df, extra)


def t_paired_vs_v1(ctx):
    ctx.reset()
    methods = [m for m in ("v2_cheap", "v2_main", "v2_antipure", "v2_final", "photoguard", "mist") if m in ctx.an.methods()]
    df = _paired_rows(ctx, [(m, "v1") for m in methods] or [("v2_final", "v1")])
    return ctx.table("paired_vs_v1", "Paired differences vs v1 (no purifier, per editor)", df, PAIRED_NOTE)


def t_paired_vs_noise(ctx):
    ctx.reset()
    methods = [m for m in ("v1", "photoguard", "mist", "v2_cheap", "v2_main", "v2_antipure", "v2_final") if m in ctx.an.methods()]
    comps = [(m, n) for m in (methods or ["v2_final"]) for n in ("noise_uniform", "noise_gauss_psnr")]
    df = _paired_rows(ctx, comps)
    return ctx.table("paired_vs_noise", "Paired differences vs the noise controls (no purifier, per editor)", df, PAIRED_NOTE)


def _gate_df(g: A.GateResult, cols_map):
    rows = []
    for r in g.rows:
        row = {}
        for k, fn in cols_map.items():
            row[k] = fn(r)
        rows.append(row)
    return pd.DataFrame(rows)


def _f(key, nd=2):
    return lambda r: fmt_num(r.get(key), nd) if r.get(key) is not None else NOT_RUN


def _fci(key, lo, hi, nd=2, signed=False):
    return lambda r: fmt_ci(r.get(key), r.get(lo), r.get(hi), nd, signed) if r.get(key) is not None and math.isfinite(r.get(key, float("nan"))) else "n/a"


def t_e3(ctx):
    ctx.reset()
    g = ctx.gates["E3"]
    rows = []
    for r in g.rows:
        pur = r["purifier"]
        c = ctx.an.cell(Spec("clean", pur))
        ctx.cell(Spec("v1", pur))
        ctx.cell(Spec("v1", "none"))
        rows.append({"purifier": pur, "v1 PSR (no purifier)": fmt_num(r["psr_none"]),
                     "v1 PSR (purified) [95% CI]": fmt_ci(r["psr"], r["psr_lo"], r["psr_hi"]),
                     "drop [95% CI]": fmt_ci(r["diff"], r["diff_lo"], r["diff_hi"], 2, True), "p": fmt_p(r["p"]),
                     "relative drop": fmt_num(r["rel_drop"]), "in decision": "yes" if r["core"] else "no",
                     "clean + purifier PSR (control)": A.cell_text(c, "psr")})
    df = pd.DataFrame(rows) if rows else pd.DataFrame([{"purifier": NOT_RUN}])
    return ctx.table("e3", f"E3: v1 under purification (SD1.5 s0.55). Decision: {g.verdict}", df,
                     "Decision rule: " + g.rule + " " + " ".join(g.notes) + " Paired drop = PSR(purified) - PSR(no purifier), same images.")


def t_e4(ctx):
    ctx.reset()
    g = ctx.gates["E4"]
    for r in g.rows:
        ctx.cell(Spec(ctx.gp.e4_method, r["purifier"]))
        ctx.cell(Spec("v1", r["purifier"]))
    rows = [{"purifier": r["purifier"] + ("" if r["primary"] else " (held-out, info)"),
             "v1 PSR": fmt_num(r["psr_v1"]) if r["psr_v1"] is not None else NOT_RUN,
             f"{ctx.gp.e4_method} PSR": fmt_num(r["psr_new"]) if r["psr_new"] is not None else NOT_RUN,
             "diff [95% CI]": fmt_ci(r["diff"], r["diff_lo"], r["diff_hi"], 2, True) if r["psr_v1"] is not None else NOT_RUN,
             "p": fmt_p(r["p"]), "ratio [95% CI]": fmt_ci(r["ratio"], r["ratio_lo"], r["ratio_hi"], 2) if r["psr_v1"] is not None else NOT_RUN,
             "gate (>= %gx)" % ctx.gp.e4_factor: r["status"]} for r in g.rows]
    return ctx.table("e4", f"E4: {ctx.gp.e4_method} vs v1 under cheap purifiers (SD1.5 s0.55). Gate: {g.verdict}", pd.DataFrame(rows),
                     "Gate: " + g.rule + " " + " ".join(g.notes))


def t_e5(ctx):
    ctx.reset()
    g = ctx.gates["E5"]
    rows = []
    for r in g.rows:
        ctx.cell(Spec(r["method"], r["purifier"]))
        d = ctx.an.paired(Spec(r["method"], r["purifier"]), Spec("v1", r["purifier"]))
        rows.append({"method": r["method"], "purifier": r["purifier"], "PSR (no purifier)": fmt_num(r["psr_none"]) if r["psr_none"] is not None else NOT_RUN,
                     "PSR (purified)": fmt_num(r["psr"]) if r["psr"] is not None else NOT_RUN,
                     "ratio purified/none [95% CI]": fmt_ci(r["ratio"], r["ratio_lo"], r["ratio_hi"]) if r["psr"] is not None else NOT_RUN,
                     "minus v1 under same purifier [95% CI]": fmt_ci(d["mean_diff"], d["ci_lo"], d["ci_hi"], 2, True) if d else NOT_RUN,
                     "p": fmt_p(d["p_wilcoxon"]) if d else "", "gate": r["status"]})
    return ctx.table("e5", f"E5: {ctx.gp.e5_method} (+ v2_final) under held-out purifiers (SD1.5 s0.55). Gate: {g.verdict}",
                     pd.DataFrame(rows), "Gate: " + g.rule + " " + " ".join(g.notes) +
                     " diffpure_* share the SD1.5 UNet with the one-step training purifier (gray-box, V2_PLAN R17); gridpure/restore_swin2sr do not.")


def t_e6(ctx):
    ctx.reset()
    g = ctx.gates["E6"]
    rows = []
    for r in g.rows:
        pur = r["purifier"]
        ctx.cell(Spec(ctx.gp.e6_method, pur))
        ctx.cell(Spec(ctx.gp.e6_ref, pur))
        rows.append({"purifier": pur, f"{ctx.gp.e6_ref} PSR": fmt_num(r.get("psr_ref")) if r.get("psr_ref") is not None else NOT_RUN,
                     f"{ctx.gp.e6_method} PSR": fmt_num(r.get("psr_new")) if r.get("psr_new") is not None else NOT_RUN,
                     "diff [95% CI]": fmt_ci(r.get("diff"), r.get("diff_lo"), r.get("diff_hi"), 2, True) if r.get("diff") is not None else NOT_RUN,
                     "p": fmt_p(r.get("p", float("nan"))), "verdict": r["status"]})
    return ctx.table("e6", f"E6: anti-purification terms vs v2_main. Go/no-go: {g.verdict}", pd.DataFrame(rows), "Rule: " + g.rule)


def t_e7(ctx):
    pur = ["none", "jpeg_q50", "vae_rt", "diffpure_0.1", "diffpure_0.2"]
    ms = ["abl_vis_only", "abl_den_only", "abl_no_sem", "abl_no_pur", "v2_main"]
    t = _matrix(ctx, "e7", "E7 ablation: PSR by variant and purifier (SD1.5 s0.55; v2_main = both phases, with L_sem)", ms, pur,
                lambda m, p: Spec(m, p),
                extra="abl_vis_only and abl_den_only also drop L_sem (beta=0) to isolate the phases; compare abl_no_sem vs v2_main for L_sem, "
                      "abl_no_pur vs v2_main for the purifiers in the loop.")
    return t


def t_e7_editors(ctx):
    cells = [c for c in EDITOR_CELLS if c[1] in (A.HEAD_EDITOR, "ipadapter_faceid", "ip2p", "sd_inpaint") and c[2] in ("0.55", "", "1")]
    ms = ["abl_vis_only", "abl_den_only", "abl_no_sem", "abl_no_pur", "v2_main"]
    return _matrix(ctx, "e7_editors", "E7 ablation: PSR by variant and editor, no purifier", ms, cells,
                   lambda m, c: Spec(m, "none", c[1], c[2]), col_label=lambda c: c[0])


def t_e7_paired(ctx):
    ctx.reset()
    comps = [("abl_den_only", "abl_vis_only"), ("v2_main", "abl_vis_only"), ("v2_main", "abl_den_only"), ("v2_main", "abl_no_sem"),
             ("abl_no_sem", "abl_no_pur")]
    cells = [c for c in EDITOR_CELLS if c[1] in (A.HEAD_EDITOR, "ipadapter_faceid", "ip2p", "sd_inpaint") and c[2] in ("0.55", "", "1")]
    df = _paired_rows(ctx, comps, cells, ("none", "jpeg_q50", "vae_rt", "diffpure_0.1", "diffpure_0.2"))
    # non-headline editors only exist without purifier
    df = df[~((df["purifier"] != "none") & (df["editor"] != "SD1.5 s0.55"))]
    return ctx.table("e7_paired", "E7: paired differences between ablation variants", df, PAIRED_NOTE)


def t_e8(ctx):
    ctx.reset()
    rows = []
    for m in ("clean", "v1", "v2_final"):
        for name, ed, st in EDITOR_CELLS:
            if ed not in (A.HEAD_EDITOR, "ipadapter_faceid") or (ed == A.HEAD_EDITOR and st != "0.55"):
                continue
            sn, sa = Spec(m, "none", ed, st), Spec(m, "adaptive_ae", ed, st)
            pr = ctx.an.paired(sa, sn)
            rows.append({"method": lab(m), "editor": name, "PSR (no purifier)": ctx.txt(sn), "PSR (adaptive AE)": ctx.txt(sa),
                         "change [95% CI]": fmt_ci(pr["mean_diff"], pr["ci_lo"], pr["ci_hi"], 2, True) if pr else NOT_RUN,
                         "p": fmt_p(pr["p_wilcoxon"]) if pr else "", "ArcFace sim (adaptive AE)": ctx.txt(sa, "sim", 3)})
    return ctx.table("e8", "E8: adaptive attacker (autoencoder purifier trained on clean/protected pairs of disjoint images)",
                     pd.DataFrame(rows), "Adaptive mode, reported separately from the non-adaptive headline. The attacker is weak by construction "
                     "(120 disjoint images, 128 px patches, ~1 GPU-hour): the drop is a LOWER BOUND on the damage an adaptive attacker can do.")


def _eps_methods(ctx):
    """{eps_255: method}: epsN variants + v2_final (its own eps)."""
    ed = ctx.edits
    out = {}
    for m, g in ed[ed["method"] != "clean"].groupby("method"):
        e = g["eps_255"].dropna()
        import re
        if re.fullmatch(r"(v2_final_)?eps_?\d+", m):
            k = int(re.findall(r"\d+", m)[-1])
            out[k] = m
        elif m == "v2_final" and len(e):
            out.setdefault(int(round(e.median())), m)
    return dict(sorted(out.items()))


def t_e9(ctx):
    ctx.reset()
    em = _eps_methods(ctx)
    rows = []
    inv = ctx.inv.set_index(["method", "split"]) if len(ctx.inv) else None
    for k, m in em.items():
        r = {"eps (x/255)": k, "method": m}
        if inv is not None and (m, "test") in inv.index:
            iv = inv.loc[(m, "test")]
            r["PSNR dB [95% CI]"] = fmt_ci(iv.psnr_mean, iv.psnr_lo, iv.psnr_hi, 2)
            r["SSIM"] = fmt_num(iv.ssim_mean, 3)
        else:
            r["PSNR dB [95% CI]"] = r["SSIM"] = NOT_RUN
        for p in ("none", "jpeg_q75", "diffpure_0.1"):
            r[f"PSR SD1.5 {p}"] = ctx.txt(Spec(m, p))
        r["PSR FaceID none"] = ctx.txt(Spec(m, "none", "ipadapter_faceid", "0.55"))
        rows.append(r)
    df = pd.DataFrame(rows) if rows else pd.DataFrame([{"eps (x/255)": NOT_RUN}])
    return ctx.table("e9", "E9: epsilon sweep (invisibility vs robustness); eps=4 is v2_final", df,
                     "Plot PSR against PSNR with scripts/make_figures.py (psr_vs_psnr). Epsilon on the standard 0-255 scale.")


def t_invisibility_methods(ctx):
    ctx.reset()
    rows = []
    for r in ctx.inv.itertuples(index=False):
        rows.append({"method": r.method, "split": r.split, "n images": r.n_images, "eps (x/255)": fmt_num(r.eps_255, 0),
                     "PSNR dB [95% CI]": fmt_ci(r.psnr_mean, r.psnr_lo, r.psnr_hi, 2), "min PSNR": fmt_num(r.psnr_min),
                     "below 38 dB": r.n_below_psnr, "SSIM [95% CI]": fmt_ci(r.ssim_mean, r.ssim_lo, r.ssim_hi, 4),
                     "below 0.95": r.n_below_ssim, "LPIPS(VGG) [95% CI]": fmt_ci(r.lpips_mean, r.lpips_lo, r.lpips_hi, 3),
                     "source": r.source})
    df = pd.DataFrame(rows) if rows else pd.DataFrame([{"method": NOT_RUN}])
    note = ("Protected vs clean image, published uint8 PNG convention. Requirement: PSNR >= 38 dB and SSIM >= 0.95 on every image. "
            "n images and CI as in the cell; 'source' says whether the numbers come from the protection run (protect.csv) or the edit rows.")
    return A.Table("invisibility_methods", "Invisibility of protected images per method (PSNR, SSIM, LPIPS)", df, ctx.inv.copy(), note, ctx.synthetic)


def t_messy(ctx):
    cols = [("SD1.5 s0.55", A.HEAD_EDITOR, "0.55"), ("IP-Adapter FaceID", "ipadapter_faceid", "0.55")]
    return _matrix(ctx, "messy", "PSR on the 20 messy photos (pose / lighting / pre-compression), no purifier",
                   ["clean", "noise_uniform", "noise_gauss_psnr", "photoguard", "mist", "v1", "v2_main", "v2_final"], cols,
                   lambda m, c: Spec(m, "none", c[1], c[2], "messy"), col_label=lambda c: c[0],
                   extra="Messy photos: one rotated prompt per image, 20 images only, so intervals are wide.")


def claims(ctx):
    """Numbers for the sentence 'DiffShield lowers ArcFace identity retention of edits by A points versus a noise control
    and B points over v1, including after [held-out purifiers]; an adaptive attacker reduces this to C'."""
    out = {}
    rows = []
    ctx.reset()
    for pur in ["none"] + [p for p in HELDOUT if p in ("jpeg_other", "downup_other", "diffpure_0.1", "diffpure_0.2", "diffpure_0.3")]:
        for ref in (("noise_uniform", "noise_gauss_psnr", "v1") if pur == "none" else ("v1",)):   # noise controls: no purifier
            a, b = Spec(ref, pur), Spec("v2_final", pur)
            ctx.cell(a)
            ctx.cell(b)
            pr = ctx.an.paired(a, b, "sim")           # ref - v2_final: positive = v2_final removes more identity
            pp = ctx.an.paired(b, a, "psr")
            if pr is None:
                rows.append({"purifier": pur, "v2_final vs": ref, "pairs": 0, "ArcFace sim reduction, points [95% CI]": NOT_RUN,
                             "p": "", "PSR gain [95% CI]": NOT_RUN})
                continue
            rows.append({"purifier": pur, "v2_final vs": ref, "pairs": pr["n_pairs"],
                         "ArcFace sim reduction, points [95% CI]": fmt_ci(100 * pr["mean_diff"], 100 * pr["ci_lo"], 100 * pr["ci_hi"], 1),
                         "p": fmt_p(pr["p_wilcoxon"]),
                         "PSR gain [95% CI]": fmt_ci(pp["mean_diff"], pp["ci_lo"], pp["ci_hi"], 2, True) if pp else "n/a"})
            if pur == "none":
                out[{"noise_uniform": "A", "noise_gauss_psnr": "A_gauss", "v1": "B"}[ref]] = rows[-1]["ArcFace sim reduction, points [95% CI]"] + " points"
    ad = ctx.cell(Spec("v2_final", "adaptive_ae"))
    out["C"] = (A.cell_text(ad, "psr") + " PSR") if ad else NOT_RUN
    df = pd.DataFrame(rows) if rows else pd.DataFrame([{"purifier": NOT_RUN}])
    t = ctx.table("claims", "Claim numbers: v2_final vs noise controls and vs v1 (SD1.5 s0.55)", df,
                  "ArcFace sim reduction = mean over images of sim(reference edit) - sim(v2_final edit), in points (x100). "
                  "Sentence template: 'Under the stated threat model, DiffShield lowers ArcFace identity retention of edits by A points versus a "
                  "noise control and B points over v1, including after [held-out purifiers]; an adaptive attacker reduces PSR to C.' " + PAIRED_NOTE)
    return t, out


def t_gates(ctx):
    rows = []
    for k, g in ctx.gates.items():
        rows.append({"gate": g.id, "what": g.title, "verdict": g.verdict, "rule (operationalisation)": g.rule,
                     "notes": " ".join(g.notes)})
    return A.Table("gates", "Gate evaluation (reported honestly; a missed gate is reported, not changed)", pd.DataFrame(rows), None,
                   "Gate values are the handoff's suggestions. E3 cut-offs and purifier sets are analyst choices stated in the rule column; "
                   "change them with --gate_param. 'DEGENERATE' = the doubling rule is meaningless (V2_PLAN R20): ask the user.", ctx.synthetic)


def t_e10(ctx):
    h = ctx.human
    if not h:
        df = pd.DataFrame([{"task": "A (2AFC: which image has noise)", "result": NOT_RUN}, {"task": "B (same person?)", "result": NOT_RUN}])
        return A.Table("e10", "E10: human study", df, None, "Run scripts/make_human_study.py, collect the rater CSVs, then scripts/analyze_human_study.py.", ctx.synthetic)
    rows = []
    a = h.get("task_a") or {}
    if a:
        rows.append({"task": "A: 2AFC accuracy (50% = cannot see the noise)", "n raters": a.get("n_raters"), "n trials": a.get("n_trials"),
                     "result": fmt_ci(a.get("accuracy"), a.get("ci_lo"), a.get("ci_hi"), 2), "p vs 50%": fmt_p(a.get("p_vs_chance"))})
    for cond, b in (h.get("task_b") or {}).items():
        rows.append({"task": f"B: 'same person' rate, {cond}", "n raters": b.get("n_raters"), "n trials": b.get("n_trials"),
                     "result": fmt_ci(b.get("same_rate"), b.get("ci_lo"), b.get("ci_hi"), 2), "p vs 50%": ""})
    return A.Table("e10", "E10: human study" + (" (PILOT: fewer than 10 raters)" if h.get("pilot") else ""), pd.DataFrame(rows), None,
                   "Binomial (Clopper-Pearson) CIs over pooled trials; trials of one rater are not independent, see the per-rater analysis in summary.json.",
                   ctx.synthetic)


TABLES = {"coverage": t_coverage, "e0": t_e0, "e1": t_e1, "e2": t_e2, "e3": t_e3, "e4": t_e4, "e5": t_e5, "e6": t_e6,
          "e7": t_e7, "e7_editors": t_e7_editors, "e7_paired": t_e7_paired, "e8": t_e8, "e9": t_e9, "e10": t_e10,
          "headline": t_headline, "headline_full": t_headline_full, "per_editor": t_per_editor, "per_prompt": t_per_prompt,
          "messy": t_messy, "invisibility_methods": t_invisibility_methods, "paired_vs_v1": t_paired_vs_v1,
          "paired_vs_noise": t_paired_vs_noise, "claims": lambda c: claims(c)[0], "gates": t_gates}


# ================================================================================================== numbers
def collect_numbers(ctx: Ctx) -> dict:
    n = {}
    an = ctx.an
    for m in HEAD_METHODS:
        for p in HEAD_PURS + ["jpeg_q90", "jpeg_other", "downup_other", "vae_rt", "diffpure_0.05", "diffpure_0.3", "restore_swin2sr", "gridpure", "adaptive_ae"]:
            st = an.cell(Spec(m, p))
            if st is None:
                continue
            n[f"psr.{m}.{p}"] = A.cell_text(st, "psr")
            n[f"psr_pt.{m}.{p}"] = fmt_num(st["psr"]["mean"])
            n[f"sim.{m}.{p}"] = A.cell_text(st, "sim", 3)
            n[f"n.{m}.{p}"] = st["n_images"]
    for g in ctx.gates.values():
        n[f"gate.{g.id}"] = g.verdict
    for r in ctx.inv.itertuples(index=False):
        if r.split == "test":
            n[f"inv.{r.method}.psnr"] = fmt_ci(r.psnr_mean, r.psnr_lo, r.psnr_hi, 2)
            n[f"inv.{r.method}.ssim"] = fmt_ci(r.ssim_mean, r.ssim_lo, r.ssim_hi, 4)
            n[f"inv.{r.method}.lpips"] = fmt_ci(r.lpips_mean, r.lpips_lo, r.lpips_hi, 3)
            n[f"inv.{r.method}.n"] = r.n_images
    for k, v in (ctx.thresholds.get("per_editor") or {}).items():
        n[f"threshold.{k}"] = f"{v:.3f}"
    if "pooled" in ctx.thresholds:
        n["threshold.pooled"] = f"{ctx.thresholds['pooled']:.3f}"
    _, cl = claims(ctx)
    for k, v in cl.items():
        n[f"claim.{k}"] = v
    n["n_images_reference"] = ctx.ref_n
    gp = ctx.edits[ctx.edits["purifier"] == "gridpure"]
    n["gridpure.status"] = (f"ran: {gp['image_id'].nunique()} images, {gp.groupby(['method', 'editor']).ngroups} method/editor cells"
                            if len(gp) else "not run (no rows; documented limitation unless it ran in a later job)")
    try:
        import yaml
        b = (yaml.safe_load((REPO / "config" / "config.yaml").read_text(encoding="utf-8")) or {}).get("budget", {})
        n["budget.confirmed"] = ("confirmed on real pixels (budget.eps_confirmed=true in config/config.yaml)" if b.get("eps_confirmed")
                                 else "not yet confirmed (budget.eps_confirmed=false in config/config.yaml)")
    except Exception:
        pass
    if ctx.optuna_db and Path(ctx.optuna_db).is_file():
        try:
            od = A.read_optuna_sqlite(Path(ctx.optuna_db))
            comp = od[od["state"].astype(str).isin(["COMPLETE", "1", "TrialState.COMPLETE"])]
            n["optuna.trials"] = f"{len(comp)} completed ({int((comp['feasible'] == True).sum())} feasible)"  # noqa: E712
        except Exception:
            pass
    n["synthetic"] = "YES - SYNTHETIC DATA, NOT RESULTS" if ctx.synthetic else "no"
    if ctx.human:
        a = ctx.human.get("task_a") or {}
        if a:
            n["human.task_a"] = fmt_ci(a.get("accuracy"), a.get("ci_lo"), a.get("ci_hi"), 2)
            n["human.n_raters"] = a.get("n_raters")
            n["human.pilot"] = "PILOT" if ctx.human.get("pilot") else "full"
    return n


# ======================================================================================================= CLI
def build_ctx(combined: Path, thresholds_path=None, scope="per_editor", n_boot=10000, gate_params=(), human_path=None) -> Ctx:
    combined = Path(combined)
    edits = A.typed_edits(A.read_results_csv(combined / "edits.csv"))
    protect = A.typed_protect(A.read_results_csv(combined / "protect.csv")) if (combined / "protect.csv").is_file() else None
    tp = Path(thresholds_path) if thresholds_path else combined / "psr_thresholds.json"
    thr = json.loads(tp.read_text()) if tp.is_file() else {}
    hp = Path(human_path) if human_path else REPO / "results" / "human_study" / "summary.json"
    human = json.loads(hp.read_text()) if hp.is_file() else None
    return Ctx(edits, protect, thr, scope, n_boot, human, A.parse_gate_params(gate_params))


def run_tables(ctx: Ctx, out: Path, which, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for name in which:
        t = TABLES[name](ctx)
        A.write_table(t, out)
        done[name] = str(out / f"{name}.md")
        log(f"wrote {name}")
    for gid, g in ctx.gates.items():
        (out / f"gate_{gid}.md").write_text(A.gate_to_markdown(g), encoding="utf-8")
    nums = collect_numbers(ctx)
    (out / "numbers.json").write_text(json.dumps(nums, indent=2, default=str), encoding="utf-8")
    (out / "gates.json").write_text(json.dumps({k: g.as_dict() for k, g in ctx.gates.items()}, indent=2, default=str), encoding="utf-8")
    return done


def render_files(files, tables_dir: Path, out: Path, log=print) -> None:
    nums_p = tables_dir / "numbers.json"
    numbers = json.loads(nums_p.read_text()) if nums_p.is_file() else {}
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        res, missing = A.render_template(text, tables_dir, numbers)
        if numbers.get("synthetic", "no") != "no":
            res = f"> **{A.SYNTH_BANNER}**\n\n" + res
        (out / Path(f).name).write_text(res, encoding="utf-8")
        log(f"rendered {f} -> {out / Path(f).name} ({len(missing)} placeholders still PENDING)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_id", default=None, help="Phase 0 mode: a results/<run_id> folder with metrics.csv")
    ap.add_argument("--output_root", default=str(REPO / "results"))
    ap.add_argument("--table", default="all", help="Phase 0: invisibility. Combined mode: all | comma list of table names")
    ap.add_argument("--combined", default=None, help="folder written by scripts/ingest_results.py")
    ap.add_argument("--out", default=None, help="output folder (combined mode: results/tables; render mode: reports/rendered)")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--scope", default="per_editor", choices=["per_editor", "pooled"])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--gate_param", nargs="*", default=[], metavar="k=v")
    ap.add_argument("--human", default=None, help="results/human_study/summary.json")
    ap.add_argument("--optuna_db", default=None, help="Optuna SQLite file (only used for the trial count in numbers.json)")
    ap.add_argument("--synthetic_demo", default=None, help="DIR: generate a SYNTHETIC tree, ingest it and build tables (software check)")
    ap.add_argument("--render", nargs="*", default=None, help="template files to fill from --tables_dir")
    ap.add_argument("--tables_dir", default=str(REPO / "results" / "tables"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.render is not None:
        render_files(a.render, Path(a.tables_dir), Path(a.out or REPO / "reports" / "rendered"))
        return 0
    if a.synthetic_demo:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ingest_results", REPO / "scripts" / "ingest_results.py")
        ing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ing)
        root = Path(a.synthetic_demo)
        print(A.write_synthetic_tree(root, 24))
        ing.ingest(root, root / "combined", log=print)
        a.combined, a.out = str(root / "combined"), a.out or str(root / "tables")
        a.n_boot = min(a.n_boot, 1000)
    if a.run_id and not a.combined:
        rd = Path(a.output_root) / a.run_id
        t = invisibility(rd)
        (rd / "tables").mkdir(exist_ok=True)
        t.to_csv(rd / "tables" / "invisibility.csv", index=False)
        cols = list(t.columns)
        md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
        (rd / "tables" / "invisibility.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
        return 0
    combined = Path(a.combined or REPO / "results" / "combined")
    if not (combined / "edits.csv").is_file():
        print(f"{combined / 'edits.csv'} not found: run scripts/ingest_results.py first", file=sys.stderr)
        return 2
    ctx = build_ctx(combined, a.thresholds, a.scope, a.n_boot, a.gate_param, a.human)
    ctx.optuna_db = a.optuna_db
    which = list(TABLES) if a.table in ("all", "") else [t for t in a.table.split(",") if t]
    bad = [t for t in which if t not in TABLES]
    if bad:
        print(f"unknown tables {bad}; known: {sorted(TABLES)}", file=sys.stderr)
        return 2
    out = Path(a.out or REPO / "results" / "tables")
    run_tables(ctx, out, which)
    print(f"tables in {out}" + ("  [SYNTHETIC DATA]" if ctx.synthetic else ""))
    g = ctx.gates
    print("gates: " + "; ".join(f"{k}: {v.verdict}" for k, v in g.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
