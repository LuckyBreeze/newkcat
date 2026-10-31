"""End-to-end scientific integrity gates for the replacement-model runner."""

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from newkcat.sequence.chemistry import dynamic_spans
from newkcat.sequence.model import SequenceModelConfig
from newkcat.sequence import runner
from newkcat.sequence.tokenizer import ReactionTokenizer


SR = "[CH3]{[CH2]|[CH]}{-|=}{[OH]|[O]}"


def _write_rows(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_prepared_fixture(path, rows):
    """Seal synthetic prepared rows; converter chemistry has separate tests."""
    _write_rows(path, rows)
    provenance = path.parent / "reaction_provenance.jsonl"
    provenance.write_text('{}\n', encoding="utf-8")
    report = {
        "schema_version": 1, "n": len(rows), "samples_written": True,
        "rows_dropped": 0, "tokenizer": ReactionTokenizer().metadata(),
        "samples_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
    }
    (path.parent / "coverage.json").write_text(json.dumps(report), encoding="utf-8")


def experiment(tmp_path, encoder="sr_smiles", pooling="mean", poison_test=False,
               missing_reaction="error"):
    rows = [{
        "sample_id": f"sample{i}", "sequence": "M"+"ACDEFG"[i]*5,
        "smiles": "CCO", "log10_kcat": str(i/4 - 0.5),
        "split": "train" if i < 3 else "val" if i < 5 else "test",
        "group_id": f"cluster{i}", "reaction_status": "ok",
        "reaction_smiles": "CCO>>CC=O", "sr_smiles": SR,
        "edit_spans": json.dumps(dynamic_spans(SR)),
    } for i in range(6)]
    if poison_test:
        rows[-1].update(log10_kcat="do_not_read", sr_smiles="do_not_parse",
                        reaction_smiles="do_not_parse", edit_spans="do_not_decode")
    samples = tmp_path / "samples.csv"
    _write_prepared_fixture(samples, rows)
    ids = [row["sample_id"] for row in rows][::-1]
    rng = np.random.default_rng(32)
    protein = rng.normal(size=(6, 1024)).astype(np.float32)
    if poison_test:
        protein[0] = np.nan
    features = tmp_path / "features.npz"
    extra = {}
    if encoder == "catapro_features":
        extra = {"molt5": rng.normal(size=(6, 768)).astype(np.float32),
                 "maccs": rng.integers(0, 2, size=(6, 167), dtype=np.int8)}
        extra["maccs"][:, 0] = 0
    np.savez(features, sample_ids=np.asarray(ids, dtype="U"), protein=protein, **extra)
    config = {
        "data": {"samples": "samples.csv", "features": "features.npz",
                 "missing_reaction": missing_reaction},
        "model": {"chemical_encoder": encoder, "pooling": pooling, "d_model": 16,
                  "nhead": 2, "num_layers": 1, "dim_feedforward": 32,
                  "head_dim": 16, "dropout": 0.0, "max_length": 128},
        "training": {"seed": 668, "batch_size": 2, "epochs": 2,
                     "patience": 1, "lr": 0.001, "num_threads": 1,
                     "device": "cpu", "deterministic": True},
        "output_dir": "run",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return runner.load_config(config_path), rows


@pytest.mark.parametrize("encoder,pooling", [
    ("catapro_features", "mean"), ("reaction_smiles", "mean"),
    ("sr_smiles", "mean"), ("sr_smiles", "edit_conditioned"),
])
def test_train_all_comparison_arms_without_reading_test(tmp_path, encoder, pooling):
    config, _ = experiment(tmp_path, encoder, pooling, poison_test=True)
    result = runner.fit(config)
    assert result["status"] == "completed"
    assert result["train_count"] == 3 and result["val_count"] == 2
    assert result["epochs_completed"] <= 2 and result["best_epoch"] in {1, 2}
    assert result["test_evaluated"] is False
    assert result["coverage"] == {"train": {"catapro_features" if encoder == "catapro_features"
                                            else "ok": 3},
                                   "val": {"catapro_features" if encoder == "catapro_features"
                                          else "ok": 2}}
    checkpoint = runner.load_checkpoint(result["checkpoint"])
    assert checkpoint["split_ids"]["test"] == ["sample5"]
    assert checkpoint["chemical_encoder"] == encoder
    assert checkpoint["model_config"]["pooling"] == pooling
    if encoder == "catapro_features":
        assert checkpoint["initialization"]["kind"] == "cached_catapro_features"
    else:
        assert checkpoint["initialization"]["pretrained"] is False
    assert not (config.output_dir / "test_predictions.csv").exists()


def test_locked_checkpoint_replays_validation_metrics(tmp_path):
    config, _ = experiment(tmp_path)
    result = runner.fit(config)
    checkpoint = Path(result["checkpoint"])
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replay = runner.infer(config, checkpoint, output_dir=tmp_path / "replay", split="val")
    assert replay["n"] == 2 and replay["weights_changed"] is False
    assert replay["metrics"] == result["best_validation"]
    assert before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (tmp_path / "replay" / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["sample3", "sample4"]


def test_checkpoint_architecture_and_fallback_policy_override_yaml(tmp_path):
    config, rows = experiment(tmp_path, missing_reaction="substrate")
    result = runner.fit(config)
    rows[-1].update(reaction_status="fallback_substrate", reaction_smiles="", sr_smiles="",
                    edit_spans="[]")
    updated = tmp_path / "updated.csv"
    _write_prepared_fixture(updated, rows)
    contradictory = replace(
        config, samples=updated, chemical_encoder="reaction_smiles", missing_reaction="error",
        model=SequenceModelConfig(d_model=32, nhead=4, num_layers=2, max_length=512),
    )
    replay = runner.infer(contradictory, result["checkpoint"],
                          output_dir=tmp_path / "test_out", split="test")
    assert replay["chemical_encoder"] == "sr_smiles"
    assert replay["model_config"]["d_model"] == 16
    assert replay["coverage"] == {"fallback_substrate": 1}


def test_prediction_ignores_bad_labels_but_evaluation_rejects_them(tmp_path):
    config, rows = experiment(tmp_path)
    result = runner.fit(config)
    rows[-1]["log10_kcat"] = "unknown"
    changed = tmp_path / "unlabeled.csv"
    _write_prepared_fixture(changed, rows)
    config = replace(config, samples=changed)
    prediction = runner.infer(config, result["checkpoint"], output_dir=tmp_path / "predict",
                              split="test", evaluate=False)
    assert prediction["metrics_computed"] is False and "metrics" not in prediction
    with (tmp_path / "predict" / "predictions.csv").open() as handle:
        predicted_rows = list(csv.DictReader(handle))
    assert "target" not in predicted_rows[0]
    with pytest.raises(ValueError, match="target"):
        runner.infer(config, result["checkpoint"], output_dir=tmp_path / "evaluate", split="test")


def test_heldout_evaluation_rejects_training_overlap(tmp_path):
    config, rows = experiment(tmp_path)
    result = runner.fit(config)
    copied = dict(rows[0], split="test")
    evaluation = tmp_path / "overlap.csv"
    _write_prepared_fixture(evaluation, [copied])
    with pytest.raises(ValueError, match="Held-out test overlaps"):
        runner.infer(replace(config, samples=evaluation), result["checkpoint"],
                     output_dir=tmp_path / "overlap_out", split="test")


def test_nonfinite_checkpoint_is_rejected_before_inference(tmp_path):
    config, _ = experiment(tmp_path)
    result = runner.fit(config)
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    for tensor in checkpoint["model_state"].values():
        if tensor.is_floating_point():
            tensor.flatten()[0] = float("nan")
            break
    poisoned = tmp_path / "poisoned.pt"
    torch.save(checkpoint, poisoned)
    with pytest.raises(ValueError, match="finite real"):
        runner.infer(config, poisoned, output_dir=tmp_path / "bad_out")
    assert not (tmp_path / "bad_out").exists()


def test_training_cannot_overwrite_inputs_even_when_overwrite_enabled(tmp_path):
    config, _ = experiment(tmp_path)
    before = config.samples.read_bytes()
    with pytest.raises(ValueError, match="contain input"):
        runner.fit(replace(config, output_dir=tmp_path), overwrite=True)
    assert config.samples.read_bytes() == before


def test_prediction_cannot_overwrite_checkpoint_directory(tmp_path):
    config, _ = experiment(tmp_path)
    result = runner.fit(config)
    with pytest.raises(ValueError, match="contain input"):
        runner.infer(config, result["checkpoint"], output_dir=config.output_dir, overwrite=True)


def test_duplicate_configuration_keys_and_random_freeze_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data: {}\ndata: {}\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.load_config(path)
    config, _ = experiment(tmp_path)
    raw = yaml.safe_load(config.config_path.read_text())
    raw["freeze_encoder"] = True
    config.config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="random encoder"):
        runner.load_config(config.config_path)


def test_edit_pooling_is_not_allowed_for_unannotated_control(tmp_path):
    config, _ = experiment(tmp_path)
    with pytest.raises(ValueError, match="edit_conditioned requires"):
        runner.load_config(config.config_path, encoder="reaction_smiles", pooling="edit_conditioned")


def test_training_output_is_not_silently_reused(tmp_path):
    config, _ = experiment(tmp_path)
    runner.fit(config)
    with pytest.raises(FileExistsError, match="not empty"):
        runner.fit(config)


@pytest.mark.parametrize("artifact", ["samples", "provenance"])
def test_modified_prepared_artifacts_are_rejected_before_training(tmp_path, artifact):
    config, rows = experiment(tmp_path)
    if artifact == "samples":
        rows[0]["sr_smiles"] = "CC"
        rows[0]["edit_spans"] = "[]"
        _write_rows(config.samples, rows)
    else:
        (tmp_path / "reaction_provenance.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="provenance changed|samples.*changed"):
        runner.fit(config)
    assert not config.output_dir.exists()


def test_reaction_models_require_preparation_but_baseline_does_not(tmp_path):
    config, _ = experiment(tmp_path, encoder="catapro_features")
    (tmp_path / "coverage.json").unlink()
    (tmp_path / "reaction_provenance.jsonl").unlink()
    runner.validate(config, split="train")
    with pytest.raises(ValueError, match="require sequence prepare artifacts"):
        runner.validate(replace(config, chemical_encoder="sr_smiles"), split="train")
