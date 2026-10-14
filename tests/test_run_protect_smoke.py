import json
import subprocess
import sys

import pandas as pd
import pytest

import scripts.run_protect as rp  # noqa: E402  (scripts is importable via repo root on sys.path)


def _args(synth_dir, out_root, run_id, *extra):
    return ["--method", "v1", "--input_dir", str(synth_dir), "--output_root", str(out_root), "--run_id", run_id,
            "--backend", "tiny", "--device", "cpu", "--no_lpips", "--set", "budget.iters=2",
            "--set", "data.image_size=64", *extra]


def test_run_protect_smoke_resume_and_shards(synth_dir, tmp_path):
    out = tmp_path / "results"
    assert rp.main(_args(synth_dir, out, "full")) == 0
    d = out / "full"
    for f in ("config.json", "git_hash.txt", "versions.json", "env.json", "seed.txt", "command.txt",
              "summary.json", "gpu_time.json", "metrics.csv"):
        assert (d / f).exists(), f
    df = pd.read_csv(d / "metrics.csv")
    assert len(df) == 3 and len(list((d / "images").glob("*.png"))) == 3
    assert (df.png_maxabs_u8 <= 4).all()  # 8/255 on [-1,1] == 4 gray levels
    assert json.loads((d / "config.json").read_text())["_method"] == "v1"

    # resume adds no rows
    assert rp.main(_args(synth_dir, out, "full", "--resume")) == 0
    assert len(pd.read_csv(d / "metrics.csv")) == 3

    # shards + merge == unsharded (per-image deterministic seeds)
    assert rp.main(_args(synth_dir, out, "sharded", "--shard", "0/2")) == 0
    assert rp.main(_args(synth_dir, out, "sharded", "--shard", "1/2")) == 0
    import scripts.merge_shards as ms

    assert ms.main(["--run_dir", str(out / "sharded")]) == 0
    a = pd.read_csv(d / "metrics.csv").sort_values("image_id").reset_index(drop=True)
    b = pd.read_csv(out / "sharded" / "metrics.csv").sort_values("image_id").reset_index(drop=True)
    assert list(a.image_id) == list(b.image_id) and len(b) == 3
    assert (a.seed == b.seed).all()
    assert (a.png_maxabs_u8 == b.png_maxabs_u8).all()


def test_resume_from_copies_and_skips(synth_dir, tmp_path):
    out = tmp_path / "results"
    assert rp.main(_args(synth_dir, out, "a", "--limit", "2")) == 0
    assert rp.main(_args(synth_dir, out, "b", "--resume_from", str(out / "a"))) == 0
    df = pd.read_csv(out / "b" / "metrics.csv")
    assert len(df) == 3 and df.image_id.is_unique
