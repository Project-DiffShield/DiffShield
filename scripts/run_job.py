"""Run one or more jobs / sessions from kaggle/jobs.yaml sequentially in this Kaggle session.

python scripts/run_job.py --jobs S06,S15 [--jobs_file kaggle/jobs.yaml] [--work /kaggle/working]
       [--job_shard K/N] [--resume_from DIR] [--n_gpus N] [--set key=val ...] [--force] [--dry_run]

Each job writes everything into <work>/<JOB_ID>[_sKofN]/ (results under .../results/<run_id>/), a DONE marker, a
kaggle_session.json (wall-clock seconds, for the GPU ledger) and a zip <work>/<JOB_ID>[_sKofN]_outputs.zip
(without prepared images/caches, except cache/clean_edits and cache/arcface).
Jobs are resumable: a job with a DONE marker is skipped unless --force; --resume_from points at the folder of an
earlier session's output (layout <resume_from>/<JOB_ID>/results/<run_id>); protection and evaluation resume per row.

jobs.yaml schema (see kaggle/jobs.yaml):
  macros:  name: {defaults: {k: v}, steps: [...]}     # steps may contain <<k>> placeholders (whole-string => object)
  jobs:
    <JOB_ID>:
      title, accelerator (informational), extends: <other JOB_ID> (optional; params merged, steps inherited if absent)
      params: {name: value}            # overridable with --set name=value; used as {name} in steps
      smoke_params: {name: value}      # applied on top when param smoke is true (smoke also sets n_images=5, iters=5)
      steps:
        - macro: name
          with: {k: v}                 # fills <<k>> in the macro; when/when_not on the macro line apply to all its steps
        - name: str
          run: [argv...]               # {py} {code} {out} {results} {cfg} {splits_dir} {run_id} {job_id} {n_gpus} + params
          sharded: true                # one process per GPU, CUDA_VISIBLE_DEVICES=g, "--shard K/N" appended
          workers: true                # one process per GPU with "--worker-id w<g>" appended (Optuna)
          optional: true               # failure does not abort the job (but the job is then not marked DONE)
          soft: true                   # failure is ignored completely (checks, optional reports)
          when: <param> | [params]     # run only if all those params are truthy ("when_not" for the inverse)
          timeout_min: 180             # kill the step after N minutes (allow_timeout: true => counts as success)
          run_id: <str>                # resume key for {resume_arg}
          eval_jobs: {defaults: {...}, jobs: [{name, tier, method, split, ...}]}
                                       # written to a yaml file, referenced as {jobs_file} in run (scripts/evaluate.py)
          builtin: pip_install | resolve_splits | make_config | set_var
  Special argv elements (expanded in place): {extra_args} {resume_arg} {limit_args} {iters_args} {celebahq_args}
  {parsing_args}.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
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


# ------------------------------------------------------------------------------------------------- loading
def load_jobs(path):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    jobs = dict(d.get("jobs", {}))
    macros = d.get("macros") or jobs.pop("_macros", None) or {}
    jobs["_macros"] = macros
    return jobs


def subst(obj, mapping):
    if isinstance(obj, str):
        m = re.fullmatch(r"<<(\w+)>>", obj)
        if m and m.group(1) in mapping:
            return copy.deepcopy(mapping[m.group(1)])
        return re.sub(r"<<(\w+)>>", lambda mm: str(mapping[mm.group(1)]) if mm.group(1) in mapping else mm.group(0),
                      obj)
    if isinstance(obj, list):
        return [subst(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: subst(v, mapping) for k, v in obj.items()}
    return obj


def _aslist(x):
    return [] if x is None else (list(x) if isinstance(x, (list, tuple)) else [x])


def expand_steps(steps, macros):
    out = []
    for st in steps or []:
        if "macro" in st:
            body = macros[st["macro"]]
            mapping = {**(body.get("defaults") or {}), **(st.get("with") or {})}
            mapping = {k: (subst(v, mapping) if isinstance(v, str) else v) for k, v in mapping.items()}
            inner = expand_steps(subst(copy.deepcopy(body["steps"]), mapping), macros)
            for s in inner:
                for k in ("when", "when_not"):
                    if st.get(k) is not None:
                        s[k] = _aslist(s.get(k)) + _aslist(st[k])
                out.append(s)
        else:
            out.append(dict(st))
    return out


def resolve_job(jobs, jid):
    if jid not in jobs or jid == "_macros":
        raise KeyError(f"job {jid!r} not in jobs file (have {sorted(k for k in jobs if k != '_macros')})")
    j = dict(jobs[jid])
    if j.get("extends"):
        base = resolve_job(jobs, j["extends"])
        merged = dict(base)
        merged.update({k: v for k, v in j.items() if k not in ("params", "smoke_params", "steps")})
        merged["params"] = {**base.get("params", {}), **j.get("params", {})}
        merged["smoke_params"] = {**base.get("smoke_params", {}), **j.get("smoke_params", {})}
        merged["steps"] = j["steps"] if "steps" in j else base.get("steps", [])
        j = merged
    return j


# ------------------------------------------------------------------------------------------------- helpers
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


class _Safe(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def fmt(arg, vars_):
    if isinstance(arg, str):
        try:
            return arg.format(**vars_)
        except KeyError as e:
            raise KeyError(f"unknown variable {e} in step argument {arg!r}")
    return str(arg)


def fmt_safe(obj, vars_):
    """Format nested strings, leaving unknown {placeholders} (e.g. {method}) for the receiving script."""
    if isinstance(obj, str):
        return obj.format_map(_Safe({k: v for k, v in vars_.items() if isinstance(v, (str, int, float))}))
    if isinstance(obj, list):
        return [fmt_safe(x, vars_) for x in obj]
    if isinstance(obj, dict):
        return {k: fmt_safe(v, vars_) for k, v in obj.items()}
    return obj


def expand(argv, vars_, step=None):
    out = []
    for a in argv:
        if isinstance(a, list):
            out += [fmt(x, vars_) for x in a]
        elif a == "{extra_args}":
            out += [str(x) for x in (vars_.get("extra_args") or [])]
        elif a == "{resume_arg}":
            rd = vars_.get("resume_root")
            rid = fmt(step["run_id"], vars_) if step and step.get("run_id") else vars_.get("run_id")
            cand = Path(rd) / "results" / rid if rd else None
            out += ["--resume_from", str(cand)] if cand and cand.exists() else []
        elif a in ("{limit_args}", "{iters_args}", "{celebahq_args}", "{parsing_args}"):
            out += list(vars_.get(a[1:-1]) or [])
        else:
            out.append(fmt(a, vars_))
    return out


def find_celebahq(root="/kaggle/input"):
    for r, dirs, files in os.walk(root):
        if "CelebA-HQ-img" in dirs and "CelebAMask-HQ-attribute-anno.txt" in files:
            return Path(r)
    return None


def step_enabled(st, params):
    for k in _aslist(st.get("when")):
        if not params.get(k):
            return False
    for k in _aslist(st.get("when_not")):
        if params.get(k):
            return False
    return True


# ------------------------------------------------------------------------------------------------- builtins
def pip_install(code: Path, log, extras=()):
    if not (os.path.exists("/kaggle") or os.environ.get("DIFFSHIELD_FORCE_INSTALL")):
        log("pip_install: not on Kaggle, skipping (set DIFFSHIELD_FORCE_INSTALL=1 to force)")
        return 0
    want = {""} | {f"[{e}]" for e in extras}  # sections "# === INSTALL (core)" and "# === INSTALL[eval]"
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
    rc = subprocess.call(cmd)
    # S05: the Kaggle image ships torchao 0.10.0, which diffusers/peft reject (LoRA / IP-Adapter FaceID loading raises
    # ImportError "incompatible version of torchao"). We never use torchao, so remove it.
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    return rc


def b_resolve_splits(st, vars_, params, log):
    code = Path(vars_["code"])
    real = code / "data" / "splits"
    if (real / "test.txt").exists() and (real / "val.txt").exists() and not params.get("force_smoke_splits"):
        vars_["splits_dir"] = str(real)
        log(f"splits: using frozen splits in the bundle ({real})")
        return 0
    if not params.get("smoke") and not params.get("allow_smoke_splits"):
        log("splits: the frozen splits (data/splits/test.txt, val.txt, ...) are missing from this bundle. Run S03 and put "
            "its splits_out/*.txt into data/splits/ (bundle B2), or use PARAMS {'smoke': true} for a throw-away smoke run.")
        return 1
    d = Path(vars_["out"]) / "splits"
    rc = subprocess.call([sys.executable, "scripts/make_smoke_splits.py", "--out_dir", str(d)], cwd=code)
    vars_["splits_dir"] = str(d)
    log(f"splits: SMOKE-ONLY splits generated from the v1 list in {d} (no frozen splits in this bundle)")
    return rc


def b_make_config(st, vars_, params, log):
    code, out = Path(vars_["code"]), Path(vars_["out"])
    dst = out / "cfg"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(code / "config", dst)
    raw = st.get("final_yaml") or params.get("final_yaml") or ""
    fy = fmt(raw, vars_) if raw else ""
    if fy:
        if Path(fy).exists():
            shutil.copy2(fy, dst / "parts" / "final.yaml")
            log(f"config: tuned final.yaml taken from {fy}")
        elif params.get("smoke") or st.get("optional"):
            log(f"config: final_yaml {fy} not found; continuing with the bundle config (hand-set v2_final)")
        else:
            log(f"config: final_yaml {fy} not found")
            return 1
    elif (dst / "parts" / "final.yaml").exists():
        log("config: using config/parts/final.yaml from the bundle")
    else:
        log("config: no tuned final.yaml; v2_final = hand-set v2_final_default")
    vars_["cfg"] = str(dst / "config.yaml")
    return 0


def b_set_var(st, vars_, params, log):
    for k, v in (st.get("vars") or {}).items():
        vars_[k] = fmt(v, vars_)
        log(f"set {k} = {vars_[k]}")
    return 0


# ------------------------------------------------------------------------------------------------- steps
def write_eval_jobs(st, vars_, params, tag, a, suffix=""):
    spec = copy.deepcopy(st["eval_jobs"])
    defaults = spec.setdefault("defaults", {})
    n = params.get("n_images")
    if n:
        defaults.setdefault("limit", int(n))
    sets = list(defaults.get("set") or [])
    if params.get("smoke"):
        sets.append("eval.seeds=[1]")
    defaults["set"] = sets
    defaults.setdefault("config", vars_["cfg"])
    defaults.setdefault("cache_dir", "{out}/cache")
    defaults.setdefault("no_generate_clean", True)
    pa = vars_.get("parsing_args") or []
    if pa and "parsing_dir" not in defaults:
        defaults["parsing_dir"] = pa[1]
    for j in spec["jobs"]:
        sp = j.get("split", defaults.get("split", "test"))
        j.setdefault("split", sp)
        j.setdefault("split_file", "{splits_dir}/" + sp + ".txt")
        j.setdefault("source_dir", "{out}/data/" + sp)
        if str(j.get("method", "")) != "clean":
            j.setdefault("protected_dir", "{results}/prot_{method}_" + sp + "/images")
        j.setdefault("out", "{results}/" + j["name"])
        if a.resume_from:
            cand = Path(a.resume_from) / tag / "results" / j["name"]
            if cand.exists():
                j["resume_from"] = str(cand)
    spec = fmt_safe(spec, vars_)
    p = Path(vars_["out"]) / f"eval_jobs.{st['name']}{suffix}.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return p


def run_step(st, vars_, params, env_base, logdir: Path, log, tag, a):
    name = st["name"]
    b = st.get("builtin")
    if b == "pip_install":
        return pip_install(Path(vars_["code"]), log, st.get("extras", []))
    if b in ("resolve_splits", "make_config", "set_var"):
        return {"resolve_splits": b_resolve_splits, "make_config": b_make_config, "set_var": b_set_var}[b](
            st, vars_, params, log)
    n_gpus = vars_["n_gpus"]
    js = vars_["job_shard_t"]
    code = vars_["code"]
    base_argv = st["run"]
    timeout = float(st["timeout_min"]) * 60 if st.get("timeout_min") else None
    if st.get("sharded") or st.get("workers"):
        procs = []
        ng = max(n_gpus, 1)
        n_total = (js[1] if js else 1) * ng
        for g in range(ng):
            v = dict(vars_)
            if st.get("eval_jobs"):
                v["jobs_file"] = str(write_eval_jobs(st, vars_, params, tag, a, suffix=f".g{g}"))
            argv = expand(base_argv, v, st)
            if st.get("sharded"):
                k_total = (js[0] if js else 0) + (js[1] if js else 1) * g
                if n_total > 1:
                    argv += ["--shard", f"{k_total}/{n_total}"]
            else:
                argv += ["--worker-id", f"w{g}"]
            env = dict(env_base)
            if n_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            lp = logdir / f"{name}.gpu{g}.log"
            log(f"[{name}] gpu{g}: {' '.join(argv)}")
            fh = open(lp, "w", encoding="utf-8")
            procs.append((subprocess.Popen(argv, cwd=code, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, lp))
        t0, rc, timed_out = time.time(), 0, False
        while procs:
            time.sleep(5)
            if timeout and not timed_out and time.time() - t0 > timeout:
                log(f"[{name}] timeout after {timeout / 60:.0f} min: terminating workers")
                timed_out = True
                for p in procs:
                    p[0].terminate()
                time.sleep(10)
                for p in procs:
                    if p[0].poll() is None:
                        p[0].kill()
                rc = 0 if st.get("allow_timeout") else 124
            for p in list(procs):
                if p[0].poll() is not None:
                    p[1].close()
                    tail = p[2].read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
                    log(f"[{name}] {p[2].name} exited {p[0].returncode}; tail: {tail}")
                    if not timed_out:
                        rc = rc or p[0].returncode
                    procs.remove(p)
        return rc
    v = dict(vars_)
    if st.get("eval_jobs"):
        v["jobs_file"] = str(write_eval_jobs(st, vars_, params, tag, a))
    argv = expand(base_argv, v, st)
    log(f"[{name}] {' '.join(argv)}")
    lp = logdir / f"{name}.log"
    t0 = time.time()
    killed = False
    with open(lp, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(argv, cwd=code, env=env_base, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            fh.write(line)
            fh.flush()
            if st.get("echo", True):
                print(line, end="", flush=True)
            if timeout and time.time() - t0 > timeout and not killed:
                p.terminate()
                killed = True
        p.wait()
    if killed and st.get("allow_timeout"):
        return 0
    return p.returncode


def zip_dir(src: Path, dst: Path):
    keep_cache = ("cache/clean_edits", "cache/arcface")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(src).as_posix()
            if rel.startswith("data/"):
                continue
            if "cache/" in rel and not any(k in rel for k in keep_cache):
                continue
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
    user_keys = set()
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        params[k] = yaml.safe_load(v)
        user_keys.add(k)
    if params.get("smoke"):
        sp = {"n_images": 5, "iters": 5, **(job.get("smoke_params") or {})}
        for k, v in sp.items():
            if k not in user_keys:
                params[k] = v
    n_gpus = a.n_gpus if a.n_gpus is not None else count_gpus()
    run_id = f"{jid}" + (f"_s{js[0]}of{js[1]}" if js else "")
    resume_root = None
    if a.resume_from:
        cand = Path(a.resume_from) / tag
        if cand.exists():
            resume_root = str(cand)
    n_img, iters = params.get("n_images"), params.get("iters")
    cel = find_celebahq()
    parsing = (cel / "CelebAMask-HQ-mask-anno") if cel and (cel / "CelebAMask-HQ-mask-anno").is_dir() else None
    vars_ = {**params, "py": sys.executable, "code": str(REPO), "out": str(out), "results": str(results),
             "run_id": run_id, "job_id": jid, "n_gpus": n_gpus, "resume_root": resume_root,
             "job_shard_t": js, "job_shard": a.job_shard or "", "cfg": str(REPO / "config" / "config.yaml"),
             "splits_dir": str(REPO / "data" / "splits"),
             "limit_args": ["--limit", str(n_img)] if n_img else [],
             "iters_args": (["--set", f"budget.iters={iters}", "--set", f"baselines.photoguard.iters={iters}",
                             "--set", f"baselines.mist.iters={iters}"] if iters else []),
             "celebahq_args": ["--celebahq_root", str(cel)] if cel else [],
             "parsing_args": ["--parsing_dir", str(parsing)] if parsing else []}
    log(f"start: {job.get('title', '')} | gpus={n_gpus} smoke={bool(params.get('smoke'))} "
        f"n_images={n_img} iters={iters} resume={resume_root}")
    steps = expand_steps(job["steps"], jobs.get("_macros", {}))
    if a.dry_run:
        for st in steps:
            if not step_enabled(st, params):
                print("  (skipped:", st["name"], ")")
                continue
            if st.get("builtin"):
                print("  builtin", st["builtin"], {k: v for k, v in st.items() if k not in ("name", "builtin")})
            else:
                try:
                    print("  ", st["name"], ("[sharded] " if st.get("sharded") else ""),
                          ("[workers] " if st.get("workers") else ""),
                          " ".join(expand(st["run"], {**vars_, "jobs_file": "<jobs_file>"}, st)))
                except KeyError as e:
                    print("   ", st["name"], "UNRESOLVED VARIABLE", e)
        return 0
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    status, rc_job, hard_fail = {}, 0, []
    for st in steps:
        if not step_enabled(st, params):
            log(f"step {st['name']}: skipped (condition)")
            continue
        s0 = time.time()
        rc = run_step(st, vars_, params, env, out / "logs", log, tag, a)
        status[st["name"]] = {"rc": rc, "seconds": round(time.time() - s0, 1)}
        log(f"step {st['name']}: rc={rc} ({time.time() - s0:.0f}s)")
        if rc != 0 and not st.get("soft"):
            hard_fail.append(st["name"])
        if rc != 0 and not st.get("optional") and not st.get("soft"):
            rc_job = rc
            log(f"step {st['name']} failed; aborting job (rerun the same command to resume)")
            break
    sess = {"job_id": jid, "run_id": run_id, "session_seconds": time.time() - t0, "n_gpus": n_gpus,
            "steps": status, "params": dict(params), "ok": rc_job == 0 and not hard_fail,
            "failed_steps": hard_fail}
    (out / "kaggle_session.json").write_text(json.dumps(sess, indent=2, default=str))
    results.mkdir(exist_ok=True)
    shutil.copy2(out / "kaggle_session.json", results / "kaggle_session.json")
    if (results / run_id).is_dir():  # J01-style jobs: session time next to the run's gpu_time.json (GPU ledger)
        shutil.copy2(out / "kaggle_session.json", results / run_id / "kaggle_session.json")
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
        (Path(a.work) / "job_runner_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
