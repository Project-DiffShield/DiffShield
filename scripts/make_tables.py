"""Generate tables from results CSVs (never by hand).

python scripts/make_tables.py --run_id <id> [--output_root results] --table invisibility
Phase 0 provides the `invisibility` table; later tables are added by other agents in separate functions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.eval.stats import summarize  # noqa: E402

V1_SUMMARY = REPO / "data" / "v1_ref" / "stage1_summary_metrics.csv"


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--output_root", default=str(REPO / "results"))
    ap.add_argument("--table", default="invisibility", choices=["invisibility"])
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
