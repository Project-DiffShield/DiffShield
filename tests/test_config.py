import pytest
import yaml

from src.utils.config import (apply_overrides, cfg_get, config_hash, deep_merge, load_config, validate_config)


def test_real_config_loads_and_v1_preset(repo_root):
    c = load_config(repo_root / "config" / "config.yaml")
    assert c["budget"]["scale"] == "standard_0_255"
    v = load_config(repo_root / "config" / "config.yaml", method="v1")
    assert v["eot"]["k_draws"] == 1 and v["loss"]["l_str_on"] == "augmented_pair"
    assert v["budget"]["eps_255"] == 4 and v["budget"]["step_255"] == 0.5
    assert v["loss"]["alpha"] == 0.2699475469647097
    assert v["loss"]["beta"] == 0.9023875567667267
    assert v["loss"]["gamma"] == 60.52658429696642
    assert v["_method"] == "v1"


def test_overrides_parse_types(repo_root):
    c = load_config(repo_root / "config" / "config.yaml", method="v1",
                    overrides=["budget.iters=2", "runtime.device=cpu", "data.n_test=[1,2]", "x.y.z=true"])
    assert c["budget"]["iters"] == 2 and c["data"]["n_test"] == [1, 2] and c["x"]["y"]["z"] is True
    assert c["_overrides"][0] == "budget.iters=2"


def test_parts_merge_sorted(tmp_path, repo_root):
    base = yaml.safe_load(open(repo_root / "config" / "config.yaml", encoding="utf-8"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "a.yaml").write_text("methods: {v2_x: {budget: {iters: 7}}}\nfoo: 1\n")
    (tmp_path / "parts" / "b.yaml").write_text("foo: 2\n")
    c = load_config(tmp_path / "config.yaml", method="v2_x")
    assert c["foo"] == 2 and c["budget"]["iters"] == 7 and "v1" in c["methods"]


def test_hash_stable_and_validation(repo_root):
    c = load_config(repo_root / "config" / "config.yaml")
    assert config_hash(c) == config_hash(dict(reversed(list(c.items()))))
    with pytest.raises(ValueError):
        validate_config(apply_overrides(c, ["budget.scale=unit"]))
    with pytest.raises(ValueError):
        validate_config(apply_overrides(c, ["budget.eps_255=40"]))
    with pytest.raises(KeyError):
        cfg_get(c, "no.such")
    assert cfg_get(c, "no.such", 5) == 5
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
