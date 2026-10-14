"""Run one or more jobs from kaggle/jobs.yaml sequentially in this session.

python scripts/run_job.py --jobs J00,J01 [--jobs_file kaggle/jobs.yaml] [--work /kaggle/working]
       [--job_shard K/N] [--resume_from DIR] [--n_gpus N] [--set key=val ...] [--force] [--dry_run]

Each job writes everything into <work>/<JOB_ID>[_sKofN]/ (results in .../results/<run_id>/), a DONE marker, a
kaggle_session.json (wall-clock seconds, for the GPU ledger) and a zip <work>/<JOB_ID>[_sKofN]_outputs.zip.
Jobs are resumable: a job with a DONE marker is skipped unless --force; --resume_from points at the folder of an
earlier session's output (the layout <resume_from>/<JOB_ID>/results/<run_id>); protection resumes per image.

jobs.yaml schema (see kaggle/jobs.yaml):
  jobs:
    <JOB_ID>:
      title, accelerator (informational), extends: <other JOB_ID> (optional; steps/params inherited)
      params: {name: value}               # overridable with --set name=value, used as {name} in steps
      steps:
        - name: str
          run: [argv...]                    # {py} {code} {out} {results} {run_id} {job_id} {n_gpus} + params
          sharded: true|false               # one process per GPU, CUDA_VISIBLE_DEVICES=g, --shard appended
          optional: true|false              # failure does not abort the job
          when: <param name>                # step only runs if that param is truthy
          builtin: pip_install              # installs requirements.txt lines below the INSTALL marker
          cwd: code (default)
  Element "{extra_args}" expands to the list in param extra_args (default none). Element "{resume_arg}" expands to `--resume_from <dir>` when a resume dir exists, else to nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

MARKER = "# === INSTALL"


def load_jobs(path):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return d.get("jobs", {})


def resolve_job(jobs, jid):
    if jid not in jobs:
        raise KeyError(f"job {jid!r} not in jobs file (have {sorted(jobs)})")
    j = dict(jobs[jid])
    if j.get("extends"):
        base = resolve_job(jobs, j["extends"])
        merged = dict(base)
        merged.update({k: v for k, v in j.items() if k not in ("params", "steps")})
        merged["params"] = {**base.get("params", {}), **j.get("params", {})}
        merged["steps"] = j["steps"] if "steps" in j else base.get("steps", [])
        return merged
    return j


def parse_shard(s):
    if not s:
        return None
    k, n = s.split("/")
    return int(k), int(n)


def count_gpus():
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def fmt(arg, vars_):
    if isinstance(arg, str):
        try:
            return arg.format(**vars_)
        except KeyError as e:
            raise KeyError(f"unknown variable {e} in step argument {arg!r}")
    return str(arg)


def expand(argv, vars_):
    out = []
    for a in argv:
        if a == "{extra_args}":
            out += [str(x) for x in (vars_.get("extra_args") or [])]
        elif a == "{resume_arg}":
            out += ["--resume_from", vars_["resume_dir"]] if vars_.get("resume_dir") else []
        else:
            out.append(fmt(a, vars_))
    return out


def pip_install(code: Path, log, extras=()):
    if not (os.path.exists("/kaggle") or os.environ.get("DIFFSHIELD_FORCE_INSTALL")):
        log("pip_install: not on Kaggle, skipping (set DIFFSHIELD_FORCE_INSTALL=1 to force)")
        return 0
    want = {""} | {f"[{e}]" for e in extras}  # sections: "# === INSTALL (core)" and "# === INSTALL[eval]"
    lines, section = [], None
    for l in (code / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if l.startswith(MARKER):
            rest = l[len(MARKER):]
            section = rest[: rest.index("]") + 1] if rest.startswith("[") else ""
            continue
        if l.startswith("# === OPTIONAL"):
            section = None
            continue
        s = l.split("#")[0].strip()
        if section is not None and section in want and s:
            lines.append(s)
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + lines
    log("pip_install: " + " ".join(cmd))
    return subprocess.call(cmd)


def run_step(step, vars_, env_base, logdir: Path, log):
    name = step["name"]
    if step.get("builtin") == "pip_install":
        return pip_install(Path(vars_["code"]), log, step.get("extras", []))
    argv = expand(step["run"], vars_)
    n_gpus = vars_["n_gpus"]
    js = vars_["job_shard_t"]
    code = vars_["code"]
    if step.get("sharded"):
        n_total = (js[1] if js else 1) * max(n_gpus, 1)
        procs = []
        for g in range(max(n_gpus, 1)):
            k_total = (js[0] if js else 0) + (js[1] if js else 1) * g
            a = list(argv) + (["--shard", f"{k_total}/{n_total}"] if n_total > 1 else [])
            env = dict(env_base)
            if n_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            lp = logdir / f"{name}.gpu{g}.log"
            log(f"[{name}] gpu{g}: {' '.join(a)}")
            fh = open(lp, "w", encoding="utf-8")
            procs.append((subprocess.Popen(a, cwd=code, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, lp))
        rc = 0
        while procs:
            time.sleep(5)
            for p in list(procs):
                if p[0].poll() is not None:
                    p[1].close()
                    tail = p[2].read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
                    log(f"[{name}] {p[2].name} exited {p[0].returncode}; tail: {tail}")
                    rc = rc or p[0].returncode
                    procs.remove(p)
        return rc
    log(f"[{name}] {' '.join(argv)}")
    lp = logdir / f"{name}.log"
    with open(lp, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(argv, cwd=code, env=env_base, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            if step.get("echo", True):
                print(line, end="", flush=True)
        p.wait()
    return p.returncode


def zip_dir(src: Path, dst: Path):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in src.rglob("*"):
            if f.is_file() and "data/v1_inputs" not in f.as_posix():
                z.write(f, f.relative_to(src.parent))


def run_job(jid, jobs, a, session_log):
    job = resolve_job(jobs, jid)
    js = parse_shard(a.job_shard)
    tag = jid + (f"_s{js[0]}of{js[1]}" if js else "")
    work = Path(a.work)
    out = work / tag
    results = out / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}"
        print(line, flush=True)
        with open(out / "logs" / "job.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    if (out / "DONE").exists() and not a.force:
        log("DONE marker present, skipping (use --force to rerun)")
        return 0
    params = dict(job.get("params", {}))
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        params[k] = yaml.safe_load(v)
    n_gpus = a.n_gpus if a.n_gpus is not None else count_gpus()
    run_id = f"{jid}" + (f"_s{js[0]}of{js[1]}" if js else "")
    resume_dir = None
    if a.resume_from:
        cand = Path(a.resume_from) / tag / "results" / run_id
        if cand.exists():
            resume_dir = str(cand)
    vars_ = {**params, "py": sys.executable, "code": str(REPO), "out": str(out), "results": str(results),
             "run_id": run_id, "job_id": jid, "n_gpus": n_gpus, "resume_dir": resume_dir,
             "job_shard_t": js, "job_shard": a.job_shard or ""}
    log(f"start: {job.get('title', '')} | gpus={n_gpus} params={params} resume={resume_dir}")
    if a.dry_run:
        for st in job["steps"]:
            if st.get("when") and not params.get(st["when"]):
                print("  (skipped:", st["name"], ")")
                continue
            if st.get("builtin"):
                print("  builtin", st["builtin"])
            else:
                print("  ", st["name"], ("[sharded] " if st.get("sharded") else ""), " ".join(expand(st["run"], vars_)))
        return 0
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    status = {}
    rc_job = 0
    for st in job["steps"]:
        if st.get("when") and not params.get(st["when"]):
            log(f"step {st['name']}: skipped (param {st['when']} is false)")
            continue
        s0 = time.time()
        rc = run_step(st, vars_, env, out / "logs", log)
        status[st["name"]] = {"rc": rc, "seconds": round(time.time() - s0, 1)}
        log(f"step {st['name']}: rc={rc} ({time.time() - s0:.0f}s)")
        if rc != 0 and not st.get("optional"):
            rc_job = rc
            log(f"step {st['name']} failed; aborting job (rerun the same command to resume)")
            break
    sess = {"job_id": jid, "run_id": run_id, "session_seconds": time.time() - t0, "n_gpus": n_gpus,
            "steps": status, "params": params, "ok": rc_job == 0}
    (out / "kaggle_session.json").write_text(json.dumps(sess, indent=2))
    rd = results / run_id
    if rd.exists():
        shutil.copy2(out / "kaggle_session.json", rd / "kaggle_session.json")
    if rc_job == 0:
        (out / "DONE").write_text("ok\n")
    zip_dir(out, work / f"{tag}_outputs.zip")
    log(f"finished rc={rc_job} in {time.time() - t0:.0f}s -> {work / (tag + '_outputs.zip')}")
    session_log[tag] = sess
    return rc_job


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True, help="comma-separated job ids, run sequentially")
    ap.add_argument("--jobs_file", default=str(REPO / "kaggle" / "jobs.yaml"))
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--job_shard", default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--n_gpus", type=int, default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    jobs = load_jobs(a.jobs_file)
    ids = [j.strip() for j in a.jobs.split(",") if j.strip()]
    for j in ids:
        resolve_job(jobs, j)  # fail early on typos
    summary, rc = {}, 0
    for j in ids:
        r = run_job(j, jobs, a, summary)
        rc = rc or r
        if r != 0:
            print(f"job {j} failed (rc={r}); continuing with the next job id", flush=True)
    if not a.dry_run:
        Path(a.work).mkdir(parents=True, exist_ok=True)
        (Path(a.work) / "job_runner_summary.json").write_text(json.dumps(summary, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
