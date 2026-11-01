"""Exercise the real training/evaluation lifecycle with synthetic feature fixtures."""

import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from newkcat.interaction.runner import (
    fit, infer, inspect_model, load_checkpoint, load_config, with_overrides,
)


@pytest.fixture
def demo(tmp_path):
    # Use the shipped CLI demonstration so documentation and the runner agree.
    import importlib.util
    from pathlib import Path
    source = Path(__file__).parents[1] / "examples" / "make_interaction_demo.py"
    spec = importlib.util.spec_from_file_location("interaction_demo_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return load_config(module.make_demo(tmp_path / "demo"))


def test_training_keeps_all_rows_and_locks_before_separate_test(demo):
    result = fit(demo)
    assert result["train_count"] == 6 and result["val_count"] == 3
    assert result["test_evaluated"] is False
    ckpt = load_checkpoint(result["checkpoint"])
    assert ckpt["training_completed"] is True
    assert ckpt["target_transform"]["fitted_on"] == "train"
    assert set(ckpt["split_ids"]) == {"train", "val", "test"}
    eval_result = infer(demo, result["checkpoint"], output=demo.output_dir.parent / "test")
    assert eval_result["n"] == 3
    with open(eval_result["prediction_file"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["sample_id"] for r in rows] == ckpt["split_ids"]["test"]
    assert all(np.isfinite(float(r["prediction_log10_kcat"])) for r in rows)
    assert inspect_model(demo)["regressor_input_dim"] == 32 + 5 * 16


def test_test_labels_are_not_parsed_during_training(demo):
    with demo.samples.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row["split"] == "test":
            row["log10_kcat"] = "THIS_MUST_NOT_ENTER_TRAINING"
    with demo.samples.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = fit(demo)
    assert result["test_evaluated"] is False
    predicted = infer(demo, result["checkpoint"], evaluate=False,
                      output=demo.output_dir.parent / "predict")
    assert predicted["used_targets"] is False and predicted["n"] == 3
    with pytest.raises(ValueError):
        infer(demo, result["checkpoint"], output=demo.output_dir.parent / "bad_eval")


def test_seed_override_and_accumulation_count(demo):
    config = with_overrides(demo, seed=1024)
    assert config.training.seed == 1024
    result = fit(config)
    with (config.output_dir / "history.jsonl").open() as handle:
        history = [json.loads(line) for line in handle]
    # 6 examples, 1 per microbatch, 4 per accumulation: final 2 both contribute.
    assert all(row["optimizer_updates"] == 2 for row in history)
    checkpoint = load_checkpoint(result["checkpoint"])
    assert checkpoint["resolved_config"]["training"]["seed"] == 1024


def test_unfinished_checkpoint_rejected_and_existing_output_preserved(demo):
    result = fit(demo)
    with pytest.raises(FileExistsError):
        fit(demo)
    checkpoint = load_checkpoint(result["checkpoint"])
    checkpoint["training_completed"] = False
    unfinished = demo.output_dir.parent / "unfinished.pt"
    torch.save(checkpoint, unfinished)
    with pytest.raises(ValueError, match="incomplete"):
        load_checkpoint(unfinished)


def test_evaluation_rejects_changed_manifest_and_config_validation(demo):
    result = fit(demo)
    with demo.samples.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="differ"):
        infer(demo, result["checkpoint"], output=demo.output_dir.parent / "changed")
    with pytest.raises(ValueError):
        replace(demo.training, grad_accum_steps=0)
    with pytest.raises(ValueError):
        replace(demo.training, standardize_target="true")


def test_accumulation_matches_batched_gradient_when_dropout_disabled(demo):
    # A real behavioral check: accumulation must average by samples, including
    # the final partial window. Both runs produce batches of effective sizes 4,2.
    a = replace(demo, training=replace(demo.training, epochs=1))
    b = replace(demo, training=replace(demo.training, batch_size=4,
                                       grad_accum_steps=1, epochs=1),
                output_dir=demo.output_dir.parent / "batched")
    ra, rb = fit(a), fit(b)
    ca, cb = load_checkpoint(ra["checkpoint"]), load_checkpoint(rb["checkpoint"])
    for key in ca["model_state"]:
        torch.testing.assert_close(ca["model_state"][key], cb["model_state"][key],
                                   rtol=2e-3, atol=2e-5)


def test_failed_final_report_cannot_publish_completed_checkpoint(demo, monkeypatch):
    import newkcat.interaction.runner as runner
    original = runner._json_write

    def fail_summary(path, content):
        if path.name == "summary.json":
            raise OSError("simulated final-report failure")
        return original(path, content)

    monkeypatch.setattr(runner, "_json_write", fail_summary)
    with pytest.raises(OSError, match="final-report"):
        fit(demo)
    with pytest.raises(ValueError, match="incomplete"):
        load_checkpoint(demo.output_dir / "best.pt")
    with (demo.output_dir / "run.json").open() as handle:
        assert json.load(handle)["status"] == "failed"


def test_prediction_cannot_silently_change_global_feature_source(demo):
    result = fit(demo)
    # Supplying a path alone must trigger a source contract error, even before
    # trying to read it. Pure prediction may use new rows, not a different encoder.
    changed = replace(demo, global_features=demo.output_dir.parent / "different_global.npz")
    with pytest.raises(ValueError, match="global feature source"):
        infer(changed, result["checkpoint"], evaluate=False,
              output=demo.output_dir.parent / "different_predict")
