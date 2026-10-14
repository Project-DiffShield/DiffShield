"""J00: record the Kaggle environment (no GPU quota needed; works on CPU or GPU sessions).

python scripts/env_probe.py --out <dir>
Writes: pip_freeze.txt, env.json, hf_models.json (VAE weight sha256 from the Hub API; no downloads, no token),
pip_resolve.txt (whether the extra packages resolve). Nothing here needs credentials; failures are recorded, not fatal.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.runlog import collect_env, collect_versions  # noqa: E402

SD_IDS = ["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5"]
OTHER_IDS = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fr = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (out / "pip_freeze.txt").write_text(fr.stdout)
    env = collect_env()
    env["versions"] = collect_versions()
    env["python_version"] = sys.version
    try:
        import torch

        env["torch_arch_list_note"] = "arch list is empty on CPU-only sessions" if not torch.cuda.is_available() else ""
    except Exception:
        pass
    (out / "env.json").write_text(json.dumps(env, indent=2, default=str))

    hf = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for rid in SD_IDS + OTHER_IDS:
            try:
                info = api.model_info(rid, files_metadata=True)
                files = {s.rfilename: (s.lfs.sha256 if getattr(s, "lfs", None) else None)
                         for s in info.siblings
                         if s.rfilename in ("vae/diffusion_pytorch_model.safetensors", "vae/config.json",
                                            "model.safetensors", "pytorch_model.bin")}
                hf[rid] = {"resolves": True, "sha_commit": info.sha, "files": files}
            except Exception as e:
                hf[rid] = {"resolves": False, "error": repr(e)[:300]}
    except Exception as e:
        hf["_error"] = repr(e)[:300]
    if all(hf.get(r, {}).get("resolves") for r in SD_IDS):
        v = [hf[r]["files"].get("vae/diffusion_pytorch_model.safetensors") for r in SD_IDS]
        hf["_vae_sha256_identical_between_sd15_ids"] = (v[0] is not None and v[0] == v[1])
    (out / "hf_models.json").write_text(json.dumps(hf, indent=2))

    res = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", "lpips==0.1.4",
                          "kneed", "kornia"], capture_output=True, text=True)
    (out / "pip_resolve.txt").write_text(res.stdout + "\n" + res.stderr)
    print(f"env probe written to {out}")
    print("torch", env.get("torch"), "cuda", env.get("torch_cuda"), "gpus", env.get("gpus"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
