from dataclasses import replace
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from newkcat.model import ModelConfig
from newkcat.metrics import regression_metrics
import newkcat.train as training_module
from newkcat.train import (
    NonSingletonBatchSampler,
    TrainingConfig,
    _information_regime,
    _center_coverage,
    audit_centers,
    checked_mse,
    evaluate,
    fit,
    load_checkpoint,
    load_config,
    perturb_centers,
    predict,
    sha256_file,
)
from newkcat.types import GraphBatch


@pytest.mark.parametrize("size", list(range(2, 17)))
@pytest.mark.parametrize("batch_size", [2, 3, 4, 8])
def test_sampler_visits_every_row_once_without_singletons(size, batch_size):
    sampler = NonSingletonBatchSampler(size, batch_size, seed=42)
    batches = list(sampler)
    assert len(sampler) == len(batches)
    assert min(map(len, batches)) >= 2
    assert sorted(i for batch in batches for i in batch) == list(range(size))
    assert batches == list(sampler)


def test_loss_refuses_broadcast_and_handles_finite_gradients():
    with pytest.raises(ValueError, match="broadcast"):
        checked_mse(torch.ones(3), torch.ones(3, 1))
    with pytest.raises(ValueError, match="Nonfinite"):
        checked_mse(torch.tensor([float("nan")]), torch.ones(1))
    with pytest.raises(ValueError, match="overflowed"):
        checked_mse(torch.tensor([1e30]), torch.zeros(1))
    prediction = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = checked_mse(prediction, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 1}, {"epochs": 0}, {"lr": float("nan")}, {"seed": -1},
    {"allow_reference_centers": "false"}, {"num_workers": -1}, {"num_threads": 0},
])
def test_invalid_training_settings_fail(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_config_paths_relative_to_config_and_mode_override(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "data:\n  samples: inputs/s.csv\n  features: inputs/f.npz\n"
        "model:\n  mode: conditioned\ntraining:\n  num_threads: 1\n"
        "output_dir: outputs\n"
    )
    config = load_config(config_path, mode="baseline")
    assert config.samples == tmp_path / "inputs" / "s.csv"
    assert config.output_dir == tmp_path / "outputs"
    assert config.model["mode"] == "baseline"
    config_path.write_text(config_path.read_text().replace("num_threads", "num_thread"))
    with pytest.raises(ValueError, match="Unknown training"):
        load_config(config_path)


def test_unsafe_yaml_tags_are_rejected_without_execution(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:builtins.print ['must never execute']\n")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)


def _audit_batch():
    return GraphBatch(
        atom_features=torch.zeros(2, 4, 1), edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_features=torch.zeros(0, 1), distances=torch.zeros(2, 4, 4, dtype=torch.long),
        atom_mask=torch.tensor([[True, True, True, False], [True, True, True, True]]),
        center_scores=torch.tensor([[0.1, 0.2, 0.9, 0.0], [0.0, 0.2, 0.8, 1.0]]),
        center_available=torch.ones(2, dtype=torch.bool), protein=torch.zeros(2, 1024),
        molt5=torch.zeros(2, 768), maccs=torch.zeros(2, 167), ec2=torch.zeros(2, dtype=torch.long),
        sample_ids=("first", "second"),
    )


def test_center_permutation_preserves_mass_and_ignores_batch_order():
    batch = _audit_batch()
    result = perturb_centers(batch, "permuted", seed=5)
    assert torch.equal(result.center_scores[~batch.atom_mask], torch.zeros(1))
    assert torch.allclose(result.center_scores.sum(1), batch.center_scores.sum(1))
    reversed_batch = replace(
        batch, center_scores=batch.center_scores.flip(0), atom_mask=batch.atom_mask.flip(0),
        sample_ids=tuple(reversed(batch.sample_ids)),
    )
    reversed_result = perturb_centers(reversed_batch, "permuted", seed=5)
    assert torch.equal(reversed_result.center_scores.flip(0), result.center_scores)
    zero = perturb_centers(batch, "zero")
    assert torch.count_nonzero(zero.center_scores) == 0
    assert not zero.center_available.any()
    assert np.isfinite(result.center_scores.numpy()).all()


def test_ec_and_reference_information_require_explicit_declarations():
    model = ModelConfig(atom_dim=1, bond_dim=1, mode="conditioned")
    predicted = SimpleNamespace(center_uses_ec=True, has_reference_centers=False)
    with pytest.raises(ValueError, match="EC-informed"):
        _information_regime(model, [predicted], False)
    ec_model = replace(model, use_ec2=True)
    assert _information_regime(ec_model, [predicted], False) == "predicted_centers_with_ec"
    reference = SimpleNamespace(center_uses_ec=False, has_reference_centers=True)
    with pytest.raises(ValueError, match="oracle"):
        _information_regime(model, [reference], False)
    assert _information_regime(model, [reference], True) == "reference_center_diagnostic"
    # Global and learned baselines never consume the center annotations.
    assert _information_regime(replace(model, mode="global"), [reference], False) == "sequence_substrate"


def _write_samples(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tiny_run(tmp_path):
    """Synthetic mechanics fixture; no biochemical performance interpretation."""
    rows = []
    for i in range(8):
        split = "train" if i < 5 else "val" if i < 7 else "test"
        rows.append({
            "sample_id": f"sample-{i}", "sequence": "M" + "A" * (i + 1) + "K",
            "smiles": "CCO", "log10_kcat": 0.15 * i if split != "test" else "DO_NOT_READ_TEST",
            "split": split, "ec": ("1.1.1.1" if i % 2 else "2.2.1.1") if i < 5 else "9.9.9.9",
        })
    _write_samples(tmp_path / "samples.csv", rows)
    rng = np.random.default_rng(17)
    arrays = {
        "sample_ids": np.array([row["sample_id"] for row in rows]),
        "protein": rng.normal(0, 0.1, (8, 1024)).astype(np.float32),
        "molt5": rng.normal(0, 0.1, (8, 768)).astype(np.float32),
        "maccs": np.zeros((8, 167), dtype=np.float32),
    }
    arrays["protein"][-1] = np.nan  # fitting must never validate/use this held-out row
    np.savez(tmp_path / "features.npz", **arrays)
    annotations = [{
        "sample_id": row["sample_id"], "mapped_smiles": "[CH3:1][CH2:2][OH:3]",
        "scores": {"1": 0.0, "2": 0.1, "3": 0.9}, "kind": "predicted",
        "uses_ec": True, "source": "synthetic-test-not-biological",
    } for row in rows]
    (tmp_path / "centers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in annotations))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "data": {"samples": "samples.csv", "features": "features.npz", "centers": "centers.jsonl"},
        "model": {"mode": "conditioned", "hidden_dim": 8, "num_layers": 1,
                  "local_dim": 4, "dropout": 0.0, "use_ec2": True, "ec_dim": 4},
        "training": {"epochs": 4, "patience": 1, "min_delta": 1e6, "batch_size": 2,
                     "seed": 11, "num_threads": 1},
        "output_dir": "run",
    }))
    return load_config(config_path), rows, arrays


def test_fit_isolated_from_test_then_locked_checkpoint_eval_and_prediction(tmp_path):
    config, rows, arrays = _tiny_run(tmp_path)
    summary = fit(config)
    checkpoint_path = config.output_dir / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    original_hash = sha256_file(checkpoint_path)
    assert summary["train_count"] == 5  # final singleton was merged, not dropped
    assert summary["val_count"] == 2
    assert not summary["test_evaluated"]
    assert summary["best_epoch"] in {1, 2} and summary["epochs_completed"] == 2
    history = [json.loads(line) for line in (config.output_dir / "history.jsonl").read_text().splitlines()]
    assert checkpoint["best_val_rmse"] == min(row["val"]["rmse"] for row in history)
    assert set(summary["center_coverage"]) == {"train", "val"}
    assert summary["center_coverage"]["train"] == {
        "n": 5, "missing": 0, "predicted": 5, "reference": 0, "active_nonzero": 5,
        "observed_all_zero": 0, "ec_informed": 5,
    }
    assert checkpoint["ec_vocab"] == {"1.1": 1, "2.2": 2}
    assert checkpoint["source_hashes"]["samples_sha256"] == sha256_file(config.samples)
    assert checkpoint["split_ids"]["test"] == ["sample-7"]
    resolved = load_config(config.output_dir / "config.resolved.yaml")
    assert resolved.model == config.model
    assert resolved.features == config.features
    # An evaluation YAML cannot silently replace the trained architecture or EC vocabulary.
    inference_config = replace(config, model={"mode": "baseline", "use_ec2": False})
    validation = evaluate(inference_config, checkpoint_path, split="val", output_dir=tmp_path / "val")
    assert validation["model_config"] == checkpoint["model_config"]
    assert validation["metrics"] == summary["best_validation"]
    assert set(validation["center_coverage"]) == {"val"}
    # Existing artifacts remain protected, including when train is called again.
    with pytest.raises(FileExistsError):
        fit(config)
    assert sha256_file(checkpoint_path) == original_hash
    # Fixed-checkpoint center tests do not rewrite the checkpoint.
    audit = audit_centers(config, checkpoint_path, output_dir=tmp_path / "audit")
    assert set(audit["interventions"]) == {"original", "zero", "permuted"}
    assert not audit["weights_changed"]
    assert set(audit["center_coverage"]) == {"val"}
    assert audit["interventions"]["original"]["rmse"] == summary["best_validation"]["rmse"]
    assert sha256_file(checkpoint_path) == original_hash
    # Explicit test evaluation sees the deliberately corrupt test features and fails.
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate(config, checkpoint_path, output_dir=tmp_path / "bad-test")
    # Prediction accepts a missing target; no fake observed values or metrics are emitted.
    rows[-1]["log10_kcat"] = ""
    _write_samples(config.samples, rows)
    arrays["protein"][-1] = 0
    np.savez(config.features, **arrays)
    result = predict(config, checkpoint_path, split="test", output_dir=tmp_path / "prediction")
    assert result["n"] == 1 and result["metrics_computed"] is False
    assert "observed_log10_kcat" not in (tmp_path / "prediction" / "predictions.csv").read_text()
    rows[-1]["log10_kcat"] = -0.5
    _write_samples(config.samples, rows)
    held_out = evaluate(config, checkpoint_path, output_dir=tmp_path / "held-out")
    assert held_out["metrics"]["n"] == 1
    assert held_out["metrics"]["r2"] is None
    assert not held_out["oracle_diagnostic"]
    assert set(held_out["center_coverage"]) == {"test"}
    # External manifests cannot relabel a trained sequence as a held-out test enzyme.
    external = {**rows[0], "sample_id": "external", "split": "test"}
    _write_samples(tmp_path / "external.csv", [external])
    np.savez(tmp_path / "external.npz", sample_ids=np.array(["external"]),
             protein=arrays["protein"][:1], molt5=arrays["molt5"][:1], maccs=arrays["maccs"][:1])
    leaked = replace(config, samples=tmp_path / "external.csv", features=tmp_path / "external.npz", centers=None)
    with pytest.raises(ValueError, match="sequence_hashes"):
        evaluate(leaked, checkpoint_path, output_dir=tmp_path / "leaked-test")
    # Damaged provenance and weights fail before constructing an inference model.
    corrupted = {**checkpoint, "source_hashes": {**checkpoint["source_hashes"], "samples_sha256": "bad"}}
    torch.save(corrupted, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="source hashes"):
        load_checkpoint(tmp_path / "corrupt.pt")
    weights = dict(checkpoint["model_state"])
    key = next(name for name, value in weights.items() if value.is_floating_point())
    weights[key] = torch.full_like(weights[key], float("nan"))
    torch.save({**checkpoint, "model_state": weights}, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="finite real"):
        load_checkpoint(tmp_path / "corrupt.pt")


def test_min_delta_controls_patience_without_discarding_better_checkpoint(tmp_path, monkeypatch):
    config, _, _ = _tiny_run(tmp_path)
    # First RMSE 1.0 sets the patience anchor. The 0.9 improvement is below
    # min_delta=0.5, so patience expires, but epoch 2 must still be saved.
    config = replace(config, training=replace(config.training, min_delta=0.5, patience=1))
    errors = iter([1.0, 0.9, 0.9])

    def fixed_validation(model, loader, device, **kwargs):
        target = np.array([row["log10_kcat"] for row in loader.dataset.records])
        prediction = target + next(errors)
        return {"sample_ids": loader.dataset.sample_ids, "target": target.tolist(),
                "prediction": prediction.tolist(), "metrics": regression_metrics(target, prediction)}

    monkeypatch.setattr(training_module, "collect_predictions", fixed_validation)
    summary = fit(config)
    checkpoint = load_checkpoint(config.output_dir / "best.pt")
    assert summary["epochs_completed"] == 2
    assert summary["best_epoch"] == checkpoint["epoch"] == 2
    assert checkpoint["best_val_rmse"] == pytest.approx(0.9)


def test_coverage_distinguishes_missing_from_observed_zero_annotations():
    dataset = SimpleNamespace(
        records=[{"split": "val"}] * 3,
        items=[
            {"center_kind": "missing", "center_available": False, "center_scores": np.zeros(2), "center_uses_ec": False},
            {"center_kind": "predicted", "center_available": True, "center_scores": np.zeros(2), "center_uses_ec": True},
            {"center_kind": "reference", "center_available": True, "center_scores": np.ones(2), "center_uses_ec": False},
        ],
    )
    assert _center_coverage(dataset) == {"val": {
        "n": 3, "missing": 1, "predicted": 1, "reference": 1,
        "active_nonzero": 1, "observed_all_zero": 1, "ec_informed": 1,
    }}


def test_missing_checkpoint_provenance_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": 1, "model_config": {}}, path)
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(path)
