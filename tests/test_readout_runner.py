"""Data integrity, validation locking, and accumulation checks for R ablations."""

import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml

from newkcat.readout import runner
from newkcat.readout.cli import main
from newkcat.sequence.data import collate_sequences
from newkcat.sequence.tokenizer import ReactionTokenizer


def _prepared(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = path.parent / "reaction_provenance.jsonl"
    provenance.write_text('{}\n', encoding="utf-8")
    (path.parent / "coverage.json").write_text(json.dumps({
        "schema_version": 1, "n": len(rows), "samples_written": True,
        "rows_dropped": 0, "tokenizer": ReactionTokenizer().metadata(),
        "samples_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
    }), encoding="utf-8")


def experiment(tmp_path, *, model=None, poison_test=False):
    rows = [{
        "sample_id": f"sample{i}", "sequence": "M" + "ACDEFGHIK"[i] * 5,
        "smiles": "CCO", "log10_kcat": str(i / 4 - 0.5),
        "split": "train" if i < 5 else "val" if i < 7 else "test",
        "group_id": f"cluster{i}", "reaction_status": "reaction_only",
        "reaction_smiles": "CCO>>CC=O",
    } for i in range(9)]
    if poison_test:
        for row in rows[7:]:
            row.update(log10_kcat="do_not_read", reaction_smiles="do_not_parse")
    samples = tmp_path / "samples.csv"
    _prepared(samples, rows)
    ids = [row["sample_id"] for row in rows][::-1]
    protein = np.random.default_rng(32).normal(size=(9, 1024)).astype(np.float32)
    if poison_test:
        protein[:2] = np.nan
    features = tmp_path / "features.npz"
    np.savez(features, sample_ids=np.asarray(ids, dtype="U"), protein=protein)
    config = {
        "data": {"samples": "samples.csv", "features": "features.npz"},
        "model": {"chemical_encoder": "reaction_smiles", "d_model": 8,
                  "nhead": 2, "num_layers": 1, "dim_feedforward": 16,
                  "head_dim": 8, "chemical_dim": 8, "dropout": 0.0, "max_length": 128,
                  **(model or {})},
        "training": {"seed": 668, "batch_size": 1, "epochs": 2,
                     "patience": 1, "lr": 0.001, "num_threads": 1,
                     "device": "cpu", "deterministic": True},
        "output_dir": "run",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return runner.load_config(path), rows


@pytest.mark.parametrize("model", [
    {}, {"protein_readout": "full", "readout": "mean_max"},
    {"protein_readout": "full", "readout": "queries", "num_queries": 2,
     "chemical_dim": None},
])
def test_complete_train_never_reads_test_and_locks_validation(tmp_path, model):
    config, _ = experiment(tmp_path, model=model, poison_test=True)
    sample_before = config.samples.read_bytes()
    feature_before = config.features.read_bytes()
    result = runner.fit(config)
    assert result["status"] == "completed"
    assert result["train_count"] == 5 and result["val_count"] == 2
    assert result["test_evaluated"] is False
    checkpoint = Path(result["checkpoint"])
    saved = runner.load_checkpoint(checkpoint)
    assert saved["kind"] == runner.CHECKPOINT_KIND
    assert saved["training_completed"] is True
    assert saved["selection"] == {"metric": "global_validation_rmse", "test_used": False}
    assert saved["split_ids"]["test"] == ["sample7", "sample8"]
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replay = runner.infer(config, checkpoint, output_dir=tmp_path / "replay", split="val")
    assert replay["metrics"] == result["best_validation"]
    assert replay["weights_changed"] is False
    assert checksum == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert config.samples.read_bytes() == sample_before
    assert config.features.read_bytes() == feature_before
    assert not (config.output_dir / "test_predictions.csv").exists()
    history = [json.loads(line) for line in (config.output_dir / "history.jsonl").read_text().splitlines()]
    assert all(row["training_rows"] == 5 and row["optimizer_steps"] == 5 for row in history)


def test_independent_test_evaluation_predicts_every_original_row(tmp_path):
    config, _ = experiment(tmp_path)
    trained = runner.fit(config)
    result = runner.infer(config, trained["checkpoint"], output_dir=tmp_path / "test")
    assert result["n"] == 2 and result["metrics_computed"] is True
    with (tmp_path / "test" / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["sample7", "sample8"]
    assert all("target" in row for row in rows)


def test_benchmark_evaluation_rejects_filtered_test_rows_but_predict_allows_new_cohort(tmp_path):
    config, rows = experiment(tmp_path)
    trained = runner.fit(config)
    folder = tmp_path / "subset"
    folder.mkdir()
    subset = folder / "samples.csv"
    _prepared(subset, rows[:-1])
    selected = replace(config, samples=subset)
    with pytest.raises(ValueError, match="exactly the locked split sample IDs and order"):
        runner.infer(selected, trained["checkpoint"], output_dir=tmp_path / "filtered_test")
    assert not (tmp_path / "filtered_test").exists()
    prediction = runner.infer(selected, trained["checkpoint"], output_dir=tmp_path / "new_cohort",
                              evaluate=False)
    assert prediction["n"] == 1 and prediction["metrics_computed"] is False


@pytest.mark.parametrize("changed", ["labels", "features"])
def test_benchmark_evaluation_rejects_changed_labels_or_features(tmp_path, changed):
    config, rows = experiment(tmp_path)
    trained = runner.fit(config)
    folder = tmp_path / "changed"
    folder.mkdir()
    if changed == "labels":
        rows[7]["log10_kcat"] = "9.25"
        samples = folder / "samples.csv"
        _prepared(samples, rows)
        changed_config = replace(config, samples=samples)
    else:
        with np.load(config.features, allow_pickle=False) as original:
            ids = original["sample_ids"]
            protein = original["protein"].copy()
        protein[0, 0] += 1
        features = folder / "features.npz"
        np.savez(features, sample_ids=ids, protein=protein)
        changed_config = replace(config, features=features)
    with pytest.raises(ValueError, match="input hashes differ"):
        runner.infer(changed_config, trained["checkpoint"], output_dir=tmp_path / "changed_test")
    assert not (tmp_path / "changed_test").exists()


def test_benchmark_evaluation_allows_byte_identical_relocated_inputs(tmp_path):
    config, _ = experiment(tmp_path)
    trained = runner.fit(config)
    folder = tmp_path / "relocated"
    folder.mkdir()
    for filename in ("samples.csv", "features.npz", "coverage.json", "reaction_provenance.jsonl"):
        shutil.copyfile(tmp_path / filename, folder / filename)
    relocated = replace(config, samples=folder / "samples.csv", features=folder / "features.npz")
    result = runner.infer(relocated, trained["checkpoint"], output_dir=tmp_path / "relocated_test")
    assert result["n"] == 2
    assert result["source_hashes"] == result["training_source_hashes"]


def test_checkpoint_rejects_mismatched_yaml_instead_of_silently_overriding(tmp_path):
    config, _ = experiment(tmp_path)
    trained = runner.fit(config)
    changed = replace(config, model=replace(config.model, protein_readout="full"))
    with pytest.raises(ValueError, match="disagrees with checkpoint"):
        runner.infer(changed, trained["checkpoint"], output_dir=tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_mid_training_checkpoint_cannot_be_evaluated(tmp_path):
    config, _ = experiment(tmp_path)
    trained = runner.fit(config)
    saved = torch.load(trained["checkpoint"], map_location="cpu", weights_only=True)
    saved["training_completed"] = False
    unfinished = tmp_path / "unfinished.pt"
    torch.save(saved, unfinished)
    with pytest.raises(ValueError, match="not locked"):
        runner.infer(config, unfinished, output_dir=tmp_path / "premature_test")
    assert not (tmp_path / "premature_test").exists()


@pytest.mark.parametrize("tamper", ["kind", "selection", "weights", "shape"])
def test_checkpoint_integrity_gates(tmp_path, tamper):
    config, _ = experiment(tmp_path)
    trained = runner.fit(config)
    saved = torch.load(trained["checkpoint"], map_location="cpu", weights_only=True)
    if tamper == "kind":
        saved["kind"] = "newkcat_sequence_regressor"
    elif tamper == "selection":
        saved["selection"]["test_used"] = True
    else:
        key = next(iter(saved["model_state"]))
        if tamper == "weights":
            saved["model_state"][key].flatten()[0] = float("nan")
        else:
            saved["model_state"][key] = saved["model_state"][key].flatten()[:1]
    path = tmp_path / "invalid.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError):
        runner.load_checkpoint(path)


def test_training_and_inference_refuse_to_overwrite_inputs(tmp_path):
    config, _ = experiment(tmp_path)
    before = config.samples.read_bytes()
    with pytest.raises(ValueError, match="contain input"):
        runner.fit(replace(config, output_dir=tmp_path), overwrite=True)
    assert config.samples.read_bytes() == before
    trained = runner.fit(config)
    with pytest.raises(ValueError, match="contain input"):
        runner.infer(config, trained["checkpoint"], output_dir=config.output_dir, overwrite=True)
    with pytest.raises(FileExistsError):
        runner.fit(config)


def test_partial_accumulation_matches_sample_weighted_large_batches(tmp_path):
    config, _ = experiment(tmp_path)
    dataset = runner._dataset(config, ["train"])

    class ProteinLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1024, 1)

        def forward(self, batch):
            return self.linear(batch.protein).squeeze(-1)

    torch.manual_seed(19)
    small = ProteinLinear()
    large = ProteinLinear()
    large.load_state_dict(small.state_dict())
    micro = [collate_sequences([dataset[index] for index in indices])
             for indices in ([0, 1], [2], [3, 4])]
    combined = [collate_sequences([dataset[index] for index in indices])
                for indices in ([0, 1, 2], [3, 4])]
    options = replace(config.training, grad_accum_steps=2, grad_clip=10000.0)
    micro_result = runner._train_epoch(small, micro, torch.optim.SGD(small.parameters(), lr=.001),
                                       options, torch.device("cpu"))
    big_result = runner._train_epoch(large, combined, torch.optim.SGD(large.parameters(), lr=.001),
                                     replace(options, grad_accum_steps=1), torch.device("cpu"))
    assert micro_result[1:] == big_result[1:] == (5, 2)
    for left, right in zip(small.parameters(), large.parameters()):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)


def test_cli_overrides_and_model_inspection_do_not_need_data(tmp_path, capsys):
    config, _ = experiment(tmp_path)
    features_before = config.features.read_bytes()
    raw = yaml.safe_load(config.config_path.read_text())
    raw["data"] = {"samples": "not-present.csv", "features": "not-present.npz"}
    config.config_path.write_text(yaml.safe_dump(raw))
    assert main(["inspect-model", "--config", str(config.config_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["data_read"] is False and inspected["parameter_count"] > 0
    assert main(["validate", "--config", str(config.config_path),
                 "--samples", str(config.samples), "--features", str(config.features),
                 "--seed", "123", "--device", "cpu", "--split", "val",
                 "--output", str(tmp_path / "validation")]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 2
    assert config.features.read_bytes() == features_before
    overridden = runner.with_overrides(config, seed=123, device="cpu", output=tmp_path / "elsewhere")
    assert overridden.training.seed == 123 and overridden.output_dir == tmp_path / "elsewhere"


@pytest.mark.parametrize("field,value", [("batch_size", 0), ("batch_size", True),
                                         ("grad_accum_steps", 0), ("grad_accum_steps", 1.5)])
def test_training_option_validation(field, value):
    with pytest.raises(ValueError):
        runner.ReadoutTrainingConfig(**{field: value})


def test_readout_configuration_rejects_representation_fallback_and_random_freeze(tmp_path):
    config, _ = experiment(tmp_path)
    assert config.training.batch_size == 1
    for changed in ({"chemical_encoder": "sr_smiles"}, {"missing_reaction": "substrate"},
                    {"freeze_encoder": True}):
        with pytest.raises(ValueError):
            replace(config, **changed)
    snapshot = config.snapshot()
    snapshot["training"]["grad_accum_steps"] = 3
    config.config_path.write_text(yaml.safe_dump(snapshot))
    restored = runner.load_config(config.config_path)
    assert restored.training.grad_accum_steps == 3
    assert asdict(restored.model) == asdict(config.model)


def test_predict_ignores_unknown_targets_but_test_overlap_is_rejected(tmp_path):
    config, rows = experiment(tmp_path)
    trained = runner.fit(config)
    rows[7]["log10_kcat"] = "unknown"
    changed = tmp_path / "predict.csv"
    _prepared(changed, rows)
    prediction = runner.infer(replace(config, samples=changed), trained["checkpoint"],
                              output_dir=tmp_path / "predict", evaluate=False)
    assert prediction["metrics_computed"] is False and "metrics" not in prediction
    overlap = tmp_path / "overlap.csv"
    _prepared(overlap, [dict(rows[0], split="test")])
    with pytest.raises(ValueError, match="Held-out test overlaps"):
        runner.infer(replace(config, samples=overlap), trained["checkpoint"],
                     output_dir=tmp_path / "overlap")
