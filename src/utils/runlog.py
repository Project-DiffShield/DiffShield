"""Run logging: every run writes its resolved config, git hash, library versions, env and seed."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import config_hash

DEFAULT_PKGS = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "kornia", "numpy",
                "pillow", "scikit-image", "scikit-learn", "lpips", "insightface", "onnxruntime-gpu",
                "optuna", "kneed", "pyyaml", "pandas", "huggingface_hub")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(phase: str, method: str, tag: str, git_hash: str) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{phase}_{method}_{tag}_{str(git_hash)[:7]}_{ts}"


def get_git_info(repo_root) -> dict:
    repo_root = Path(repo_root)
    try:
        if (repo_root / ".git").exists():
            h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True,
                               timeout=20)
            if h.returncode == 0:
                dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
                                       text=True, timeout=20).stdout.strip() != ""
                br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root,
                                    capture_output=True, text=True, timeout=20).stdout.strip()
                return {"hash": h.stdout.strip(), "dirty": dirty, "branch": br, "source": "git"}
    except Exception:
        pass
    bi = repo_root / "BUNDLE_INFO.json"
    if bi.exists():
        try:
            d = json.loads(bi.read_text(encoding="utf-8"))
            return {"hash": d.get("git_hash", "unknown"), "dirty": bool(d.get("dirty", False)),
                    "branch": d.get("branch", "unknown"), "source": "bundle"}
        except Exception:
            pass
    return {"hash": "unknown", "dirty": False, "branch": "unknown", "source": "unknown"}


def collect_versions(pkgs=DEFAULT_PKGS) -> dict:
    from importlib import metadata

    out = {}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = "not-installed"
    return out


def _nvidia_smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def collect_env() -> dict:
    env = {"python": sys.version, "platform": platform.platform(),
           "kaggle": {k: v for k, v in os.environ.items()
                      if k.startswith("KAGGLE_") and not any(s in k for s in ("KEY", "USERNAME", "TOKEN", "SECRET"))}}
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["cuda_available"] = torch.cuda.is_available()
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        env["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    except Exception as e:  # pragma: no cover
        env["torch_error"] = repr(e)
    env["nvidia_smi"] = _nvidia_smi()
    return env


class GpuTimer:
    """Context manager measuring wall time with CUDA sync; accumulates into an optional owner."""

    def __init__(self, owner=None):
        self.owner = owner
        self.seconds = 0.0
        self.peak_mem_mb = 0.0

    @staticmethod
    def _cuda():
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def __enter__(self):
        if self._cuda():
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._cuda():
            import torch

            torch.cuda.synchronize()
            self.peak_mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        self.seconds = time.perf_counter() - self._t0
        if self.owner is not None:
            self.owner.gpu_seconds_total += self.seconds
            self.owner.peak_mem_mb = max(self.owner.peak_mem_mb, self.peak_mem_mb)
        return False


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class RunContext:
    run_id: str
    dir: Path
    images_dir: Path
    cfg: dict
    seed: int
    shard: tuple = None
    gpu_seconds_total: float = 0.0
    peak_mem_mb: float = 0.0
    git: dict = field(default_factory=dict)
    cfg_hash: str = ""
    _t_start: float = field(default_factory=time.time)

    @property
    def suffix(self) -> str:
        return "" if self.shard is None else f".shard{self.shard[0]}of{self.shard[1]}"

    def _name(self, name: str) -> str:
        if not self.suffix:
            return name
        stem, _, ext = name.rpartition(".")
        return f"{stem}{self.suffix}.{ext}"

    def log(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        p = self.dir / "logs" / self._name("run.log")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def csv_path(self, csv_name: str) -> Path:
        return self.dir / self._name(csv_name)

    def append_row(self, csv_name: str, row: dict) -> None:
        p = self.csv_path(csv_name)
        exists = p.exists() and p.stat().st_size > 0
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def done_ids(self, csv_name: str, key: str = "image_id") -> set:
        p = self.csv_path(csv_name)
        if not p.exists():
            return set()
        with open(p, newline="", encoding="utf-8") as f:
            return {r[key] for r in csv.DictReader(f) if r.get(key)}

    def write_progress(self, **kw) -> None:
        kw["updated_utc"] = utc_now()
        _atomic_write(self.dir / self._name("progress.json"), json.dumps(kw, indent=2, default=str))

    def gpu_timer(self) -> GpuTimer:
        return GpuTimer(owner=self)


def start_run(cfg, run_id, output_root, script, argv, seed, shard=None, repo_root=None) -> RunContext:
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    d = Path(output_root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    images = d / "images"
    images.mkdir(exist_ok=True)

    chash = config_hash(cfg)
    cj = d / "config.json"
    if cj.exists():
        old = json.loads(cj.read_text(encoding="utf-8"))
        if config_hash(old) != chash:
            raise RuntimeError(f"{cj} exists with a different config (hash {config_hash(old)} != {chash}); "
                               "use a new run_id")
    else:
        _atomic_write(cj, json.dumps(cfg, indent=2, sort_keys=True, default=str))
        try:
            import yaml

            _atomic_write(d / "config.yaml", yaml.safe_dump(cfg, sort_keys=True))
        except Exception:
            pass
    gi = get_git_info(repo_root)
    _atomic_write(d / "git_hash.txt",
                  f"{gi['hash']}\ndirty={gi['dirty']}\nbranch={gi['branch']}\nsource={gi['source']}\n")
    _atomic_write(d / "versions.json", json.dumps(collect_versions(), indent=2))
    _atomic_write(d / "env.json", json.dumps(collect_env(), indent=2, default=str))
    _atomic_write(d / "seed.txt", f"{seed}\n")
    cmd_name = "command.txt" if shard is None else f"command.shard{shard[0]}of{shard[1]}.txt"
    _atomic_write(d / cmd_name, " ".join([sys.executable, script] + list(argv)) + "\n")
    return RunContext(run_id=run_id, dir=d, images_dir=images, cfg=cfg, seed=seed, shard=shard,
                      git=gi, cfg_hash=chash)


def finish_run(ctx: RunContext, summary: dict) -> None:
    _atomic_write(ctx.dir / ctx._name("summary.json"), json.dumps(summary, indent=2, default=str))
    gpus = collect_env().get("gpus") or [None]
    n = summary.get("n_images") or 0
    gt = {"total_gpu_seconds": ctx.gpu_seconds_total, "peak_mem_mb": ctx.peak_mem_mb,
          "wall_seconds": time.time() - ctx._t_start, "n_images": n, "gpu_name": gpus[0],
          "per_image_mean": (ctx.gpu_seconds_total / n) if n else None}
    _atomic_write(ctx.dir / ctx._name("gpu_time.json"), json.dumps(gt, indent=2, default=str))
