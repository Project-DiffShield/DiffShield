"""Analyse the E10 human study from the rater CSVs saved by the HTML forms (scripts/make_human_study.py).

  python scripts/analyze_human_study.py --study_dir results/human_study
      reads  <study_dir>/answers/*.csv  and  <study_dir>/private/answer_key.csv
      writes <study_dir>/summary.json, summary.md, per_rater.csv

Task A (2AFC "which image has noise"): accuracy = share of trials where the rater picked the protected image. Reported with
  a Clopper-Pearson 95% CI and a two-sided exact binomial test against 50% (chance), pooled over trials; because trials of one
  rater are not independent, the per-rater accuracies are also summarised (mean, bootstrap CI over raters, Wilcoxon vs 0.5).
Task B ("same person?", 4-point scale): same-person rate per condition (edit of the clean photo vs edit of the protected
  photo); 'same' = Same + Probably same, 'strict' = Same only. Pooled Clopper-Pearson CIs, pooled Fisher exact test and a
  per-rater paired comparison (Wilcoxon).
Pilot flag: fewer than --min_raters (10) raters. The actual number of raters is always reported. Raters who answered fewer than
  --min_items items of a task are listed and excluded from that task. Duplicate answers of a rater for one item: last one wins.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval import analysis as A  # noqa: E402

SAME = {"same", "probably_same"}


def load_answers(answers_dir: Path):
    frames, warns = [], []
    for p in sorted(Path(answers_dir).glob("*.csv")):
        try:
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception as e:
            warns.append(f"{p.name}: unreadable ({e!r})")
            continue
        need = {"study_version", "rater_id", "task", "item_id", "response"}
        if not need <= set(d.columns):
            warns.append(f"{p.name}: missing columns {sorted(need - set(d.columns))}, skipped")
            continue
        d["_file"] = p.name
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["study_version", "rater_id", "task", "item_id", "response", "_file"]), warns
    a = pd.concat(frames, ignore_index=True)
    a["study_version"] = a["study_version"].astype(str)
    dup = a.duplicated(["study_version", "rater_id", "task", "item_id"], keep="last")
    if dup.any():
        warns.append(f"{int(dup.sum())} duplicate answers (same rater/item); the last one is kept")
    return a[~dup].reset_index(drop=True), warns


def wilson_free_p(k, n, p0=0.5):
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(k), int(n), p0).pvalue)
    except Exception:
        return float("nan")


def analyze(answers: pd.DataFrame, key: pd.DataFrame, min_raters: int = 10, min_items: int = 1, n_boot: int = 10000, seed: int = 2026) -> dict:
    key = key.copy()
    key["version"] = key["version"].astype(str)
    out = {"n_files": int(answers["_file"].nunique()) if len(answers) else 0, "warnings": []}
    ans = answers[answers["task"].isin(["A", "B"])].copy()
    ans = ans.merge(key, left_on=["study_version", "task", "item_id"], right_on=["version", "task", "item_id"], how="left")
    unmatched = int(ans["image_id"].isna().sum())
    if unmatched:
        out["warnings"].append(f"{unmatched} answers did not match the answer key (wrong key file?)")
        ans = ans[ans["image_id"].notna()]
    meta = answers[answers["task"] == "meta"]
    out["devices"] = meta["response"].value_counts().to_dict() if len(meta) else {}
    raters = sorted(ans["rater_id"].unique())
    out["n_raters_total"] = len(raters)
    per_rater = []

    # ---------- Task A
    a = ans[ans["task"] == "A"].copy()
    cnt = a.groupby("rater_id").size()
    keep = [r for r in cnt.index if cnt[r] >= min_items]
    out["raters_incomplete_a"] = {r: int(cnt[r]) for r in cnt.index if cnt[r] < min_items}
    a = a[a["rater_id"].isin(keep)]
    if len(a):
        a["correct"] = (a["response"] == a["protected_side"]).astype(int)
        k, n = int(a["correct"].sum()), int(len(a))
        acc, lo, hi = A.binom_ci(k, n)
        pr = a.groupby("rater_id")["correct"].agg(["sum", "count"])
        pr["acc"] = pr["sum"] / pr["count"]
        m, mlo, mhi = A.boot_ci(pr["acc"].to_numpy(), n_boot, 0.95, seed)
        pw, _ = A.wilcoxon_p(pr["acc"].to_numpy() - 0.5) if len(pr) >= 6 else (float("nan"), 0)
        side = a.groupby("response")["correct"].agg(["sum", "count"])
        out["task_a"] = {"n_raters": int(len(pr)), "n_trials": n, "n_correct": k, "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                         "p_vs_chance": wilson_free_p(k, n), "rater_mean_accuracy": m, "rater_ci_lo": mlo, "rater_ci_hi": mhi,
                         "p_wilcoxon_raters_vs_0.5": pw,
                         "choice_balance": {s: int(v) for s, v in a["response"].value_counts().items()},
                         "note": "2AFC: accuracy 0.5 = the added noise cannot be seen; pooled test treats trials as independent"}
        for r, row in pr.iterrows():
            per_rater.append({"rater_id": r, "task": "A", "condition": "", "n": int(row["count"]), "value": float(row["acc"])})
    else:
        out["task_a"] = {}

    # ---------- Task B
    b = ans[ans["task"] == "B"].copy()
    cnt = b.groupby("rater_id").size()
    keep = [r for r in cnt.index if cnt[r] >= min_items]
    out["raters_incomplete_b"] = {r: int(cnt[r]) for r in cnt.index if cnt[r] < min_items}
    b = b[b["rater_id"].isin(keep)]
    out["task_b"] = {}
    if len(b):
        b["same"] = b["response"].isin(SAME).astype(int)
        b["strict"] = (b["response"] == "same").astype(int)
        for cond, g in b.groupby("condition"):
            k, n = int(g["same"].sum()), int(len(g))
            rate, lo, hi = A.binom_ci(k, n)
            ks, ns = int(g["strict"].sum()), n
            sr, slo, shi = A.binom_ci(ks, ns)
            out["task_b"][cond] = {"n_raters": int(g["rater_id"].nunique()), "n_trials": n, "n_same": k, "same_rate": rate,
                                   "ci_lo": lo, "ci_hi": hi, "strict_rate": sr, "strict_ci_lo": slo, "strict_ci_hi": shi,
                                   "responses": {s: int(v) for s, v in g["response"].value_counts().items()}}
            for r, gg in g.groupby("rater_id"):
                per_rater.append({"rater_id": r, "task": "B", "condition": cond, "n": int(len(gg)), "value": float(gg["same"].mean())})
        conds = sorted(out["task_b"])
        if {"clean_edit", "protected_edit"} <= set(conds):
            c, p = out["task_b"]["clean_edit"], out["task_b"]["protected_edit"]
            try:
                from scipy.stats import fisher_exact
                pf = float(fisher_exact([[c["n_same"], c["n_trials"] - c["n_same"]], [p["n_same"], p["n_trials"] - p["n_same"]]])[1])
            except Exception:
                pf = float("nan")
            pv = pd.DataFrame([r for r in per_rater if r["task"] == "B"]).pivot_table(index="rater_id", columns="condition", values="value")
            pv = pv.dropna()
            d = (pv["clean_edit"] - pv["protected_edit"]).to_numpy() if len(pv) else np.array([])
            dm, dlo, dhi = A.boot_ci(d, n_boot, 0.95, seed) if len(d) else (float("nan"),) * 3
            pw, _ = A.wilcoxon_p(d) if len(d) >= 6 else (float("nan"), 0)
            out["task_b"]["difference"] = {"clean_minus_protected_pooled": c["same_rate"] - p["same_rate"], "p_fisher_pooled": pf,
                                           "n_raters_paired": int(len(pv)), "rater_mean_diff": dm, "rater_ci_lo": dlo, "rater_ci_hi": dhi,
                                           "p_wilcoxon_raters": pw,
                                           "note": "a lower same-person rate for edits of protected photos = identity protection visible to humans"}
    out["n_raters"] = int(len(set(a["rater_id"]) | set(b["rater_id"]))) if (len(a) or len(b)) else 0
    out["pilot"] = out["n_raters"] < min_raters
    out["min_raters_for_full_study"] = min_raters
    out["per_rater"] = per_rater
    return out


def to_markdown(s: dict) -> str:
    L = ["# Human study (E10) summary", ""]
    L.append(f"Raters analysed: **{s['n_raters']}**" + (" (**PILOT**: fewer than %d raters; interpret with caution)" % s["min_raters_for_full_study"] if s["pilot"] else ""))
    a = s.get("task_a") or {}
    if a:
        L += ["", "## Task A: which image has noise (2AFC; 50% = cannot see the noise)", "",
              f"- trials {a['n_trials']} from {a['n_raters']} raters; accuracy {a['accuracy']:.3f} "
              f"(95% Clopper-Pearson CI [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]); exact binomial test vs 0.5: p = {A.fmt_p(a['p_vs_chance'])}",
              f"- per-rater mean accuracy {a['rater_mean_accuracy']:.3f} [{a['rater_ci_lo']:.3f}, {a['rater_ci_hi']:.3f}] (bootstrap over raters); "
              f"Wilcoxon over raters vs 0.5: p = {A.fmt_p(a['p_wilcoxon_raters_vs_0.5'])} (needs >= 6 raters)",
              f"- left/right choices: {a['choice_balance']}"]
    b = s.get("task_b") or {}
    if b:
        L += ["", "## Task B: is the edited face the same person?", "", "| condition | raters | trials | same-person rate [95% CI] | strict ('Same' only) [95% CI] |", "|---|---|---|---|---|"]
        for c, v in b.items():
            if c == "difference":
                continue
            L.append(f"| {c} | {v['n_raters']} | {v['n_trials']} | {v['same_rate']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}] | "
                     f"{v['strict_rate']:.3f} [{v['strict_ci_lo']:.3f}, {v['strict_ci_hi']:.3f}] |")
        if "difference" in b:
            d = b["difference"]
            L += ["", f"- clean-edit minus protected-edit same-person rate: pooled {d['clean_minus_protected_pooled']:+.3f} (Fisher exact p = {A.fmt_p(d['p_fisher_pooled'])}); "
                      f"per-rater paired mean {d['rater_mean_diff']:+.3f} [{d['rater_ci_lo']:+.3f}, {d['rater_ci_hi']:+.3f}] over {d['n_raters_paired']} raters, "
                      f"Wilcoxon p = {A.fmt_p(d['p_wilcoxon_raters'])}"]
    if s.get("warnings"):
        L += ["", "## Warnings", ""] + [f"- {w}" for w in s["warnings"]]
    for k in ("raters_incomplete_a", "raters_incomplete_b"):
        if s.get(k):
            L.append(f"- {k}: {s[k]}")
    L += ["", "Photos are CelebA-HQ research images shown only locally to raters; they are not published."]
    return "\n".join(L) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study_dir", default=str(REPO / "results" / "human_study"))
    ap.add_argument("--answers", default=None, help="folder with rater CSVs (default <study_dir>/answers)")
    ap.add_argument("--key", default=None, help="answer key CSV (default <study_dir>/private/answer_key.csv)")
    ap.add_argument("--min_raters", type=int, default=10)
    ap.add_argument("--min_items", type=int, default=1, help="drop raters with fewer answers than this in a task")
    ap.add_argument("--n_boot", type=int, default=10000)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    sd = Path(a.study_dir)
    key_p = Path(a.key) if a.key else sd / "private" / "answer_key.csv"
    ans_dir = Path(a.answers) if a.answers else sd / "answers"
    if not key_p.is_file():
        print(f"answer key not found: {key_p}", file=sys.stderr)
        return 2
    answers, warns = load_answers(ans_dir)
    if len(answers) == 0:
        print(f"no rater CSVs in {ans_dir}: nothing to analyse (E10 not run)", file=sys.stderr)
        return 3
    key = pd.read_csv(key_p, dtype=str, keep_default_na=False)
    s = analyze(answers, key, a.min_raters, a.min_items, a.n_boot)
    s["warnings"] = warns + s["warnings"]
    pr = s.pop("per_rater")
    pd.DataFrame(pr).to_csv(sd / "per_rater.csv", index=False)
    (sd / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (sd / "summary.md").write_text(to_markdown(s), encoding="utf-8")
    print(to_markdown(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
