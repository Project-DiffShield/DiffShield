"""Generate all figures from the results CSVs (matplotlib, colour-blind-safe Okabe-Ito palette, PNG + PDF).

  python scripts/make_figures.py --combined results/combined --out results/figures [--fig all|name,name]
        [--source_dir DIR] [--edits_root DIR ...] [--clean_root DIR ...]   # only for the image figures
        [--optuna_db results/optuna/v2_final.db] [--human results/human_study/summary.json]
  python scripts/make_figures.py --synthetic_demo DIR                       # SYNTHETIC data, software check only

Figures: psr_vs_psnr (E9), psr_by_purifier, per_editor, ablation (E7), sim_distributions, before_after (needs images),
failure_gallery (needs images; failure_cases.csv is always written), optuna_history (needs the Optuna SQLite file),
human_study (needs the summary.json of scripts/analyze_human_study.py).
A figure whose input is missing is skipped and listed with the reason in figures_manifest.json (nothing is faked).
Identifiable CelebA-HQ images appear only in before_after / failure_gallery: keep those files local, never publish them.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval import analysis as A  # noqa: E402
from src.eval.analysis import Spec  # noqa: E402

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
HATCH = ["", "//", "\\\\", "xx", "..", "++", "--", "oo"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
METHOD_ORDER = ["clean", "noise_uniform", "noise_gauss_psnr", "photoguard", "mist", "v1", "v2_cheap", "v2_main",
                "v2_antipure", "v2_final"]
PLOT_PURS = ["none", "jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt", "diffpure_0.1", "diffpure_0.2"]
SHORT = {"clean": "clean", "noise_uniform": "uniform noise", "noise_gauss_psnr": "Gaussian noise", "photoguard": "PhotoGuard*",
         "mist": "Mist*", "v1": "v1", "v2_cheap": "v2 cheap", "v2_main": "v2 main", "v2_antipure": "v2 anti-purif.",
         "v2_final": "v2 final", "abl_vis_only": "L_vis only", "abl_den_only": "L_den only", "abl_no_sem": "both, no L_sem",
         "abl_no_pur": "both, no purifier"}
EDITOR_LABEL = {"sd15_img2img": "SD1.5 s0.55", "sdxl_img2img": "SDXL", "ip2p": "InstructPix2Pix", "sd_inpaint": "SD inpaint",
                "ipadapter_faceid": "IP-Adapter FaceID"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 200, "legend.frameon": False, "hatch.linewidth": 0.5})


def style(method: str, i: int):
    return OKABE[i % len(OKABE)], HATCH[i % len(HATCH)], MARKERS[i % len(MARKERS)]


def point(an: A.Analysis, spec: Spec, metric="psr"):
    st = an.cell(spec)
    if st is None or not math.isfinite(st[metric]["mean"]):
        return None
    m = st[metric]
    return m["mean"], m["lo"], m["hi"], st["n_images"]


def save(fig, out: Path, name: str, synthetic: bool, manifest: dict):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC - NOT RESULTS", color="#D55E00", fontsize=28, alpha=0.22, ha="center", va="center", rotation=20)
    files = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        files.append(p.name)
    plt.close(fig)
    manifest[name] = {"files": files, "synthetic": synthetic}


def skip(manifest, name, why):
    manifest[name] = {"skipped": why}
    print(f"skipped {name}: {why}")


def grouped_bars(ax, cats, series, ylabel="PSR", ylim=(0, 1.02)):
    """series: ordered {label: [point|None per category]}; point = (mean, lo, hi, n)."""
    k = len(series)
    w = 0.84 / max(k, 1)
    x = np.arange(len(cats))
    for i, (lab, pts) in enumerate(series.items()):
        c, h, _ = style(lab, i)
        xs, ys, lo, hi = [], [], [], []
        for j, p in enumerate(pts):
            if p is None:
                continue
            xs.append(x[j] - 0.42 + w * (i + 0.5))
            ys.append(p[0])
            lo.append(max(p[0] - p[1], 0))
            hi.append(max(p[2] - p[0], 0))
        ax.bar(xs, ys, w * 0.92, yerr=[lo, hi] if xs else None, color=c, hatch=h, edgecolor="white" if not h else "black",
               linewidth=0.4, error_kw={"elinewidth": 0.8, "capsize": 1.5}, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)


# ================================================================================================ figures
def fig_psr_vs_psnr(ctx, out, man):
    em = _eps_methods(ctx)
    if len(em) < 2:
        return skip(man, "psr_vs_psnr", "needs at least two epsilon variants (E9 not run)")
    inv = ctx.inv[ctx.inv["split"] == "test"].set_index("method")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, pur in enumerate(["none", "jpeg_q75", "diffpure_0.1"]):
        xs, ys, lo, hi, lab = [], [], [], [], []
        for k, m in em.items():
            p = point(ctx.an, Spec(m, pur))
            if p is None or m not in inv.index:
                continue
            xs.append(inv.loc[m, "psnr_mean"]); ys.append(p[0]); lo.append(p[0] - p[1]); hi.append(p[2] - p[0]); lab.append(k)
        if not xs:
            continue
        c, _, mk = style(pur, i)
        ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=mk, ms=5, lw=1.2, capsize=2, label=f"SD1.5 s0.55, purifier: {pur}")
        if i == 0:
            for x_, y_, l_ in zip(xs, ys, lab):
                ax.annotate(f"{l_}/255", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=7)
    for j, m in enumerate(["v1", "photoguard", "mist", "noise_uniform"]):
        p = point(ctx.an, Spec(m, "none"))
        if p is not None and m in inv.index:
            ax.scatter([inv.loc[m, "psnr_mean"]], [p[0]], marker=MARKERS[(j + 3) % 8], s=40, facecolor="none",
                       edgecolor=OKABE[(j + 3) % 8], label=f"{SHORT[m]} (eps 4, no purifier)", zorder=5)
    ax.axvline(38, color="gray", ls="--", lw=0.8)
    ax.text(38.05, 0.02, "PSNR 38 dB limit", fontsize=7, color="gray")
    ax.invert_xaxis()
    ax.set_xlabel("PSNR of protected image (dB, higher = less visible)")
    ax.set_ylabel("PSR (95% CI over images)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Epsilon sweep: protection vs invisibility (E9)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "psr_vs_psnr", ctx.synthetic, man)


def _eps_methods(ctx):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    return mt._eps_methods(ctx)


def fig_psr_by_purifier(ctx, out, man):
    meths = [m for m in ["noise_uniform", "photoguard", "mist", "v1", "v2_cheap", "v2_main", "v2_final"] if m in ctx.an.methods()]
    series = {SHORT[m]: [point(ctx.an, Spec(m, p)) for p in PLOT_PURS] for m in meths}
    if not any(any(v) for v in series.values()):
        return skip(man, "psr_by_purifier", "no SD1.5 s0.55 cells")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    grouped_bars(ax, PLOT_PURS, series)
    ax.set_title("PSR by purifier, SD1.5 img2img strength 0.55 (gaps = not run; * = our reimplementation)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "psr_by_purifier", ctx.synthetic, man)


def fig_per_editor(ctx, out, man):
    cells = [("SD1.5 s0.3", A.HEAD_EDITOR, "0.3"), ("SD1.5 s0.55", A.HEAD_EDITOR, "0.55"), ("SD1.5 s0.75", A.HEAD_EDITOR, "0.75"),
             ("SDXL", "sdxl_img2img", "0.55"), ("InstructPix2Pix", "ip2p", ""), ("SD inpaint", "sd_inpaint", "1"),
             ("IP-Adapter FaceID", "ipadapter_faceid", "0.55")]
    meths = [m for m in ["noise_uniform", "photoguard", "mist", "v1", "v2_cheap", "v2_main", "v2_final"] if m in ctx.an.methods()]
    series = {SHORT[m]: [point(ctx.an, Spec(m, "none", ed, st)) for _, ed, st in cells] for m in meths}
    if not any(any(v) for v in series.values()):
        return skip(man, "per_editor", "no cells")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    grouped_bars(ax, [c[0] for c in cells], series)
    ax.set_title("PSR by editor, no purifier (gaps = not run)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "per_editor", ctx.synthetic, man)


def fig_ablation(ctx, out, man):
    pur = ["none", "jpeg_q50", "vae_rt", "diffpure_0.1", "diffpure_0.2"]
    ms = [m for m in ["abl_vis_only", "abl_den_only", "abl_no_sem", "abl_no_pur", "v2_main"] if m in ctx.an.methods()]
    series = {SHORT.get(m, m) + (" (both)" if m == "v2_main" else ""): [point(ctx.an, Spec(m, p)) for p in pur] for m in ms}
    if not series or not any(any(v) for v in series.values()):
        return skip(man, "ablation", "E7 not run")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    grouped_bars(ax, pur, series)
    ax.set_title("E7 ablation: which loss terms and which phase matter (SD1.5 s0.55)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "ablation", ctx.synthetic, man)


def fig_sim_distributions(ctx, out, combined: Path, man):
    ed = ctx.edits
    base = ed[(ed["editor"] == A.HEAD_EDITOR) & (ed["strength"] == A.HEAD_STRENGTH) & (ed["purifier"] == "none") & (ed["split"] == "test")]
    groups = {}
    for m in ("clean", "noise_uniform", "v1", "v2_final"):
        s = base.loc[base["method"] == m, "arcface_sim"].dropna().to_numpy()
        if len(s):
            groups[{"clean": "clean image -> edit", "noise_uniform": "uniform noise -> edit", "v1": "v1 protected -> edit",
                    "v2_final": "v2 final protected -> edit"}[m]] = s
    imp = None
    sp, ep = combined / "arcface" / "sources.npz", combined / "arcface" / "edit_embeddings.npz"
    if sp.is_file() and ep.is_file():
        src = A.load_npz_sources([sp])
        emb = {k: v for k, v in A.load_npz_edit_embs([ep]).items() if k.startswith(A.HEAD_EDITOR + "/")}
        ids = set(base["image_id"])
        items = [(k.split("/", 2)[2], v) for k, v in emb.items() if k.split("/", 2)[2] in ids]
        srcs = {i: e for i, e in src.items() if i in ids}
        if len(srcs) > 1 and items:
            imp = A._impostor_sims(srcs, items)
    if not groups:
        return skip(man, "sim_distributions", "no SD1.5 s0.55 edits")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(-0.2, 1.0, 49)
    if imp is not None and imp.size:
        ax.hist(imp, bins=bins, density=True, histtype="stepfilled", color="#999999", alpha=0.35, label="impostors (clean edit vs other people)")
    for i, (lab, s) in enumerate(groups.items()):
        c, _, _ = style(lab, i)
        ax.hist(s, bins=bins, density=True, histtype="step", lw=1.6, color=c, label=f"{lab} (n={len(s)})")
    thr = A.threshold_for(ctx.thresholds, A.HEAD_EDITOR, ctx.scope)
    if math.isfinite(thr):
        ax.axvline(thr, color="black", ls="--", lw=1)
        ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, f"PSR threshold {thr:.2f}\n(left of line = protected)", fontsize=7)
    ax.set_xlabel("ArcFace cosine similarity to the source photo")
    ax.set_ylabel("density")
    ax.set_title("Identity retention of SD1.5 edits (strength 0.55, no purifier)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, out, "sim_distributions", ctx.synthetic, man)


# ------------------------------------------------------------------------------------- image figures
def _find_dirs(root: Path, name: str):
    out = []
    if not root.is_dir():
        return out
    for dp, dn, _ in __import__("os").walk(root):
        if "combined" in Path(dp).parts:
            dn[:] = []
            continue
        if name in dn:
            out.append(Path(dp) / name)
            dn.remove(name)
    return out


class ImageLocator:
    def __init__(self, source_dirs, edits_roots, clean_roots):
        self.source_dirs = [Path(p) for p in source_dirs]
        self.edit_roots = [Path(p) for p in edits_roots]           # <root>/edits/<method>/...
        self.clean_roots = [Path(p) for p in clean_roots]          # clean_edits dirs

    def source(self, image_id):
        return A.find_with_ext(self.source_dirs, image_id, (".png", ".jpg", ".jpeg"))

    def clean_edit(self, editor, prompt, strength, seed, image_id):
        return A.find_with_ext(self.clean_roots, A.clean_rel_path(editor, prompt, strength, seed, image_id))

    def edit(self, method, purifier, editor, prompt, strength, seed, image_id):
        rel = A.edit_rel_path(method, purifier, editor, prompt, strength, seed, image_id)
        return A.find_with_ext([r / "edits" for r in self.edit_roots], rel)

    def available(self):
        return bool(self.source_dirs) and bool(self.edit_roots)


def _show(ax, path, title=None):
    from PIL import Image
    ax.axis("off")
    ax.grid(False)
    if path is None:
        ax.text(0.5, 0.5, "image not available", ha="center", va="center", fontsize=7, transform=ax.transAxes)
    else:
        ax.imshow(Image.open(path).convert("RGB"))
    if title:
        ax.set_title(title, fontsize=7)


def _row_for(an: A.Analysis, ctx, method, image_id, purifier="none"):
    ed = ctx.edits
    r = ed[(ed["method"] == method) & (ed["image_id"] == image_id) & (ed["editor"] == A.HEAD_EDITOR) &
           (ed["strength"] == A.HEAD_STRENGTH) & (ed["purifier"] == purifier) & (ed["split"] == "test")]
    return r.sort_values(["seed", "prompt_id"])


def fig_before_after(ctx, out, loc: ImageLocator, man, n_show=6, methods=("v1", "v2_final")):
    if not loc.available():
        return skip(man, "before_after", "no source/edit images found (pass --source_dir and --edits_root; images are not in git)")
    ed = ctx.edits
    base = ed[(ed["editor"] == A.HEAD_EDITOR) & (ed["strength"] == A.HEAD_STRENGTH) & (ed["purifier"] == "none") & (ed["split"] == "test") & (ed["seed"] == 1)]
    ids = sorted(base[base["method"] == "clean"]["image_id"].unique(), key=lambda s: (len(s), s))[:n_show]
    methods = [m for m in methods if m in set(base["method"])]
    if not ids or not methods:
        return skip(man, "before_after", "no clean/protected SD1.5 s0.55 rows")
    ncol = 2 + len(methods)
    fig, axes = plt.subplots(len(ids), ncol, figsize=(1.7 * ncol, 1.7 * len(ids)), squeeze=False)
    found = 0
    for r, iid in enumerate(ids):
        rc = _row_for(ctx.an, ctx, "clean", iid)
        if len(rc) == 0:
            continue
        pid, seed = rc.iloc[0]["prompt_id"], int(rc.iloc[0]["seed"])
        _show(axes[r][0], loc.source(iid), "source" if r == 0 else None)
        sim_c = rc.iloc[0]["arcface_sim"]
        ce = loc.clean_edit(A.HEAD_EDITOR, pid, A.HEAD_STRENGTH, seed, iid)
        _show(axes[r][1], ce, f"edit of clean image\nsim {sim_c:.2f}" if r == 0 else f"sim {sim_c:.2f}")
        for c, m in enumerate(methods):
            rm = _row_for(ctx.an, ctx, m, iid)
            rm = rm[(rm["prompt_id"] == pid) & (rm["seed"] == seed)]
            p = loc.edit(m, "none", A.HEAD_EDITOR, pid, A.HEAD_STRENGTH, seed, iid)
            s = rm.iloc[0]["arcface_sim"] if len(rm) else float("nan")
            _show(axes[r][2 + c], p, (f"edit of {SHORT[m]} protected\nsim {s:.2f}" if r == 0 else f"sim {s:.2f}"))
            found += p is not None
    if found == 0:
        plt.close(fig)
        return skip(man, "before_after", "edit images not found under the given roots")
    fig.suptitle("Before / after editing (SD1.5 img2img, strength 0.55, seed 1; ArcFace sim to the source in the titles)", fontsize=8)
    fig.tight_layout()
    save(fig, out, "before_after", ctx.synthetic, man)


def failure_cases(ctx, method="v2_final", k=8):
    """Images where protection failed: highest ArcFace sim after editing (mean over seeds), SD1.5 s0.55, no purifier."""
    g = ctx.an.frame(Spec(method, "none"))
    if g is None:
        return None
    thr = A.threshold_for(ctx.thresholds, A.HEAD_EDITOR, ctx.scope)
    d = g[["sim", "psr", "no_face"]].copy()
    clean = ctx.an.frame(Spec("clean", "none"))
    if clean is not None:
        d["clean_edit_sim"] = clean["sim"]
        d["sim_drop"] = d["clean_edit_sim"] - d["sim"]
    d["threshold"] = thr
    d = d.sort_values("sim", ascending=False).head(k).reset_index()
    d.insert(0, "method", method)
    return d


def fig_failure_gallery(ctx, out, loc: ImageLocator, man, method="v2_final", k=6):
    fc = failure_cases(ctx, method, max(k, 12))
    if fc is None:
        return skip(man, "failure_gallery", f"{method} SD1.5 s0.55 no-purifier cell not run")
    fc.to_csv(out / "failure_cases.csv", index=False)
    man["failure_cases.csv"] = {"files": ["failure_cases.csv"]}
    if not loc.available():
        return skip(man, "failure_gallery", "failure_cases.csv written; images not found (pass --source_dir and --edits_root)")
    rows = fc.head(k)
    fig, axes = plt.subplots(len(rows), 3, figsize=(5.2, 1.8 * len(rows)), squeeze=False)
    found = 0
    for r, row in enumerate(rows.itertuples(index=False)):
        iid = row.image_id
        rc = _row_for(ctx.an, ctx, "clean", iid)
        pid, seed = (rc.iloc[0]["prompt_id"], int(rc.iloc[0]["seed"])) if len(rc) else ("p0", 1)
        rm = _row_for(ctx.an, ctx, method, iid)
        rm = rm[(rm["prompt_id"] == pid) & (rm["seed"] == seed)]
        s = rm.iloc[0]["arcface_sim"] if len(rm) else float("nan")
        p = loc.edit(method, "none", A.HEAD_EDITOR, pid, A.HEAD_STRENGTH, seed, iid)
        _show(axes[r][0], loc.source(iid), "source" if r == 0 else None)
        _show(axes[r][1], loc.clean_edit(A.HEAD_EDITOR, pid, A.HEAD_STRENGTH, seed, iid), "edit of clean" if r == 0 else None)
        _show(axes[r][2], p, f"edit of {SHORT[method]}\nsim {s:.2f} (mean over seeds {row.sim:.2f})")
        found += p is not None
    if found == 0:
        plt.close(fig)
        return skip(man, "failure_gallery", "failure_cases.csv written; edit images not found under the given roots")
    fig.suptitle(f"Failure cases of {method}: edits that still look like the source person", fontsize=8)
    fig.tight_layout()
    save(fig, out, "failure_gallery", ctx.synthetic, man)


# --------------------------------------------------------------------------------------------- optuna
def read_optuna_sqlite(path: Path) -> pd.DataFrame:
    """Trials of every study in an Optuna RDB (read with sqlite3, no optuna import): study, number, state, value, feasible."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tr = pd.read_sql_query("select t.trial_id, t.number, t.state, s.study_name from trials t join studies s on s.study_id = t.study_id", con)
        tv = pd.read_sql_query("select trial_id, value from trial_values where objective = 0", con)
        try:
            ua = pd.read_sql_query("select trial_id, key, value_json from trial_user_attributes where key = 'feasible'", con)
        except Exception:
            ua = pd.DataFrame(columns=["trial_id", "key", "value_json"])
    finally:
        con.close()
    d = tr.merge(tv, on="trial_id", how="left")
    if len(ua):
        d = d.merge(ua[["trial_id", "value_json"]].rename(columns={"value_json": "feasible"}), on="trial_id", how="left")
        d["feasible"] = d["feasible"].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip().lower() == "true")
    else:
        d["feasible"] = None
    return d.sort_values(["study_name", "number"]).reset_index(drop=True)


def fig_optuna(ctx, out, db: Path | None, man):
    if db is None or not Path(db).is_file():
        return skip(man, "optuna_history", "no Optuna SQLite file (pass --optuna_db results/optuna/v2_final.db)")
    d = read_optuna_sqlite(Path(db))
    d = d[d["state"].astype(str).isin(["COMPLETE", "1", "TrialState.COMPLETE"]) & d["value"].notna()]
    if len(d) == 0:
        return skip(man, "optuna_history", "no completed trials")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, (st, g) in enumerate(d.groupby("study_name")):
        x = g["number"].to_numpy()
        y = g["value"].to_numpy()
        feas = g["feasible"].map(lambda v: v is not False).to_numpy()
        c = OKABE[i % len(OKABE)]
        ax.scatter(x[feas], y[feas], color=c, s=22, label=f"{st}: feasible trial")
        if (~feas).any():
            ax.scatter(x[~feas], y[~feas], color=c, s=22, marker="x", label=f"{st}: violates PSNR/SSIM constraint")
        best = np.maximum.accumulate(np.where(feas, y, -np.inf))
        ok = np.isfinite(best)
        ax.step(x[ok], best[ok], where="post", color=c, lw=1.2)
    ax.set_xlabel("trial")
    ax.set_ylabel("objective (proxy PSR)")
    ax.set_title(f"Optuna history ({len(d)} completed trials; line = best feasible so far)")
    ax.legend(fontsize=7)
    save(fig, out, "optuna_history", ctx.synthetic, man)


def fig_human(ctx, out, man):
    h = ctx.human
    if not h or not h.get("task_a"):
        return skip(man, "human_study", "no human-study summary (E10 not run)")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    a = h["task_a"]
    ax = axes[0]
    ax.errorbar([0], [a["accuracy"]], yerr=[[a["accuracy"] - a["ci_lo"]], [a["ci_hi"] - a["accuracy"]]], fmt="o", color=OKABE[0], capsize=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1); ax.set_xticks([0]); ax.set_xticklabels(["2AFC accuracy"])
    ax.set_title(f"Task A (n raters {a.get('n_raters')}{', PILOT' if h.get('pilot') else ''})", fontsize=8)
    ax = axes[1]
    for i, (cond, b) in enumerate((h.get("task_b") or {}).items()):
        ax.errorbar([i], [b["same_rate"]], yerr=[[b["same_rate"] - b["ci_lo"]], [b["ci_hi"] - b["same_rate"]]], fmt="o", capsize=3, color=OKABE[i + 1])
    ax.set_xlim(-1, max(len(h.get("task_b") or {}), 1)); ax.set_ylim(0, 1)
    ax.set_xticks(range(len(h.get("task_b") or {}))); ax.set_xticklabels(list((h.get("task_b") or {}).keys()), rotation=15)
    ax.set_title("Task B: 'same person' rate", fontsize=8)
    save(fig, out, "human_study", ctx.synthetic, man)


FIGS = ["psr_vs_psnr", "psr_by_purifier", "per_editor", "ablation", "sim_distributions", "before_after", "failure_gallery",
        "optuna_history", "human_study"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", default=str(REPO / "results" / "combined"))
    ap.add_argument("--out", default=str(REPO / "results" / "figures"))
    ap.add_argument("--fig", default="all")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--scope", default="per_editor")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--source_dir", action="append", default=[], help="folder(s) with the clean 512 px source images")
    ap.add_argument("--edits_root", action="append", default=[], help="folder(s) that contain an edits/ directory (a job output)")
    ap.add_argument("--clean_root", action="append", default=[], help="clean_edits cache folder(s) (E0)")
    ap.add_argument("--results", default=str(REPO / "results"), help="auto-discover edits/ and clean_edits/ under this folder")
    ap.add_argument("--optuna_db", default=None)
    ap.add_argument("--human", default=None)
    ap.add_argument("--n_show", type=int, default=6)
    ap.add_argument("--synthetic_demo", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts" / "make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    if a.synthetic_demo:
        sp = importlib.util.spec_from_file_location("ingest_results", REPO / "scripts" / "ingest_results.py")
        ing = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(ing)
        root = Path(a.synthetic_demo)
        A.write_synthetic_tree(root, 24)
        ing.ingest(root, root / "combined", log=print)
        a.combined, a.out, a.n_boot = str(root / "combined"), a.out if a.out != str(REPO / "results" / "figures") else str(root / "figures"), min(a.n_boot, 1000)
    combined = Path(a.combined)
    if not (combined / "edits.csv").is_file():
        print(f"{combined / 'edits.csv'} not found: run scripts/ingest_results.py first", file=sys.stderr)
        return 2
    ctx = mt.build_ctx(combined, a.thresholds, a.scope, a.n_boot, (), a.human)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(a.results)
    loc = ImageLocator(a.source_dir, a.edits_root or [p.parent for p in _find_dirs(res, "edits")],
                       a.clean_root or _find_dirs(res, "clean_edits"))
    if not a.source_dir:
        # best effort: data/images/test, results/**/source_images
        loc.source_dirs = [p for p in (REPO / "data" / "images" / "test", REPO / "data" / "images") if p.is_dir()]
    which = FIGS if a.fig in ("all", "") else a.fig.split(",")
    man: dict = {}
    for name in which:
        try:
            if name == "psr_vs_psnr":
                fig_psr_vs_psnr(ctx, out, man)
            elif name == "psr_by_purifier":
                fig_psr_by_purifier(ctx, out, man)
            elif name == "per_editor":
                fig_per_editor(ctx, out, man)
            elif name == "ablation":
                fig_ablation(ctx, out, man)
            elif name == "sim_distributions":
                fig_sim_distributions(ctx, out, combined, man)
            elif name == "before_after":
                fig_before_after(ctx, out, loc, man, a.n_show)
            elif name == "failure_gallery":
                fig_failure_gallery(ctx, out, loc, man, "v2_final", a.n_show)
            elif name == "optuna_history":
                fig_optuna(ctx, out, Path(a.optuna_db) if a.optuna_db else None, man)
            elif name == "human_study":
                fig_human(ctx, out, man)
            else:
                print("unknown figure", name)
        except Exception as e:  # one failing figure must not block the others
            import traceback
            traceback.print_exc()
            man[name] = {"error": repr(e)}
    (out / "figures_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"figures in {out}: " + ", ".join(k for k, v in man.items() if "files" in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
