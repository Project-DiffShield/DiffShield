import json

import pytest

from src.utils.runlog import get_git_info, make_run_id, start_run


def test_start_run_writes_files_and_refuses_conflict(tmp_path):
    cfg = {"a": 1}
    ctx = start_run(cfg, "r1", tmp_path, "scripts/x.py", ["--a"], 7, repo_root=tmp_path)
    for f in ("config.json", "git_hash.txt", "versions.json", "env.json", "seed.txt", "command.txt"):
        assert (tmp_path / "r1" / f).exists()
    assert (tmp_path / "r1" / "seed.txt").read_text().strip() == "7"
    assert "source=unknown" in (tmp_path / "r1" / "git_hash.txt").read_text()
    start_run(cfg, "r1", tmp_path, "scripts/x.py", [], 7, repo_root=tmp_path)  # same config is fine
    with pytest.raises(RuntimeError):
        start_run({"a": 2}, "r1", tmp_path, "scripts/x.py", [], 7, repo_root=tmp_path)
    ctx.append_row("m.csv", {"image_id": "a", "v": 1})
    ctx.append_row("m.csv", {"image_id": "b", "v": 2})
    assert ctx.done_ids("m.csv") == {"a", "b"}


def test_bundle_info_fallback(tmp_path):
    (tmp_path / "BUNDLE_INFO.json").write_text(json.dumps({"git_hash": "abc1234", "dirty": False, "branch": "b"}))
    gi = get_git_info(tmp_path)
    assert gi["hash"] == "abc1234" and gi["source"] == "bundle"
    assert make_run_id("p0", "v1", "t", "abc1234567").startswith("p0_v1_t_abc1234_")


def test_shard_files_are_separate(tmp_path):
    c0 = start_run({"a": 1}, "r", tmp_path, "s", [], 1, shard=(0, 2), repo_root=tmp_path)
    c1 = start_run({"a": 1}, "r", tmp_path, "s", [], 1, shard=(1, 2), repo_root=tmp_path)
    c0.append_row("metrics.csv", {"image_id": "a"})
    c1.append_row("metrics.csv", {"image_id": "b"})
    assert (tmp_path / "r" / "metrics.shard0of2.csv").exists() and c0.done_ids("metrics.csv") == {"a"}
    assert c1.done_ids("metrics.csv") == {"b"}
