import pytest

from newkcat.train import load_config


def test_duplicate_yaml_key_does_not_silently_replace_method(tmp_path):
    config = tmp_path / "duplicate.yaml"
    config.write_text(
        "data:\n  samples: samples.csv\n  features: features.npz\n"
        "model:\n  mode: conditioned\n  mode: baseline\n"
    )
    with pytest.raises(ValueError, match="Duplicate YAML.*mode"):
        load_config(config)


def test_yaml_loader_does_not_construct_python_objects(tmp_path):
    config = tmp_path / "unsafe.yaml"
    config.write_text("!!python/object/apply:builtins.sum [[1, 2]]")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(config)
