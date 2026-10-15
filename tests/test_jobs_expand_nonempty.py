"""S07/S08 regression: evaluate.py defaults to `--purifier none`; a job that forgot `purifier:` in a tier without a
'none' rule (P, E8) expanded to ZERO jobs, so a whole night of purifier experiments produced empty CSVs.
Every eval job in kaggle/jobs.yaml must expand to >= 1 job, and a tier P/A/E9/E8 job must say which purifiers it wants."""
from pathlib import Path

import pytest
import yaml

from src.eval import edit_runner as er

ROOT = Path(__file__).resolve().parents[1]
JOBS = yaml.safe_load((ROOT / "kaggle" / "jobs.yaml").read_text(encoding="utf-8"))
CFG = er.load_eval_config()


def _eval_jobs():
    for sid, job in JOBS["jobs"].items():
        for st in job.get("steps", []):
            w = st.get("with") or {}
            for e in (w.get("jobs") or []):
                if isinstance(e, dict) and "tier" in e:
                    yield sid, e


def _ids(split):
    f = ROOT / "data" / "splits" / f"{split}.txt"
    return sorted(Path(l.strip()).stem for l in f.read_text().splitlines() if l.strip())


@pytest.mark.parametrize("sid,e", list(_eval_jobs()), ids=lambda x: x if isinstance(x, str) else x["name"])
def test_every_eval_job_expands_to_something(sid, e):
    if e["name"].startswith("smoke_"):
        pytest.skip("smoke jobs use throw-away splits")
    split = e.get("split", "test")
    ids = _ids(split)
    if e.get("limit"):
        ids = ids[: int(e["limit"])]
    pur = str(e.get("purifier", "none"))                       # the evaluate.py default
    pf = None if pur == "all" else [p.strip() for p in pur.split(",") if p.strip()]
    ef = [x.strip() for x in str(e["editor"]).split(",")] if e.get("editor") else None
    n = 0
    for m in [x.strip() for x in str(e["method"]).split(",")]:
        n += len(er.expand_tier(CFG["eval"], e["tier"], m, split, ids, pf, ef))
    assert n > 0, f"{sid}/{e['name']} expands to 0 jobs (tier {e['tier']}, purifier {pur!r}, split {split})"
    if e["tier"] in ("P", "A", "E9", "E8"):
        assert "purifier" in e, f"{sid}/{e['name']}: tier {e['tier']} needs an explicit purifier filter"
