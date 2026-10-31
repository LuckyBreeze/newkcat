"""Optional MLM runner checks: isolation, masking and usable encoder exports."""

import csv
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from newkcat.sequence.model import ReactionSequenceModel, SequenceModelConfig
from newkcat.sequence.pretrain import _fixed_validation, mask_tokens, pretrain
from newkcat.sequence.tokenizer import ReactionTokenizer
from newkcat.train import TrainingConfig


def tiny_config():
    return SequenceModelConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                               dropout=0.0, max_length=64, head_dim=8)


def corpus(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("CCO>>CC=O\nCBr.O>>CO.Br\nCCO>>CC=O\n", encoding="utf-8")
    val.write_text("CCC=O>>CCCO\n", encoding="utf-8")
    return train, val


def run(tmp_path, *, train=None, val=None, **kwargs):
    if train is None or val is None:
        train, val = corpus(tmp_path)
    options = dict(output_dir=tmp_path / "mlm", model_config=tiny_config(),
                   training=TrainingConfig(seed=41, epochs=2, patience=2, batch_size=2,
                                           lr=0.001, device="cpu", num_threads=1),
                   chemical_encoder="reaction_smiles")
    options.update(kwargs)
    return pretrain(train, val, **options)


def test_mlm_output_can_be_loaded_into_replacement_regressor(tmp_path):
    summary = run(tmp_path)
    assert summary["status"] == "completed"
    assert summary["epochs_completed"] == 2
    assert summary["train_texts"] == 2
    assert summary["validation_texts"] == 1
    assert not summary["test_evaluated"]
    cp = torch.load(summary["checkpoint"], map_location="cpu", weights_only=True)
    assert cp["kind"] == "newkcat_sequence_mlm"
    assert cp["chemical_encoder"] == "reaction_smiles"
    assert cp["provenance"]["train_corpus"]["duplicates_removed"] == 1
    assert cp["provenance"]["initialization"] == "random_from_scratch"
    tokenizer = ReactionTokenizer.from_state_dict(cp["tokenizer"])
    network = ReactionSequenceModel(replace(tiny_config(), pooling="edit_conditioned")).eval()
    network.encoder.load_encoder(cp["encoder"], tokenizer.fingerprint)
    ids = torch.tensor([tokenizer.encode("CCO>>CC=O")["input_ids"]], dtype=torch.long)
    batch = SimpleNamespace(protein=torch.zeros(1, 1024), input_ids=ids,
                            attention_mask=torch.ones_like(ids, dtype=torch.bool),
                            edit_mask=torch.zeros_like(ids, dtype=torch.bool))
    with torch.no_grad():
        prediction = network(batch)
    assert prediction.shape == (1,) and torch.isfinite(prediction).all()
    history = [json.loads(line) for line in (tmp_path / "mlm/history.jsonl").read_text().splitlines()]
    assert summary["best_val_loss"] == min(row["val_token_ce"] for row in history)
    assert len({row["val_supervised_tokens"] for row in history}) == 1


def test_masking_is_repeatable_and_never_supervises_specials_or_padding():
    ids = torch.tensor([[1, 41, 42, 43, 2, 0], [1, 49, 2, 0, 0, 0]])
    attention = ids != 0
    original = ids.clone()
    a, labels = mask_tokens(ids, attention, probability=0.001,
                           generator=torch.Generator().manual_seed(72))
    b, labels_b = mask_tokens(ids, attention, probability=0.001,
                             generator=torch.Generator().manual_seed(72))
    assert torch.equal(ids, original)
    assert torch.equal(a, b) and torch.equal(labels, labels_b)
    assert (labels != -100).sum(dim=1).min() >= 1
    assert (labels[ids < 4] == -100).all()
    assert torch.equal(a[ids < 4], ids[ids < 4])
    assert ((a >= 0) & (a < 98)).all()
    assert torch.equal(labels[labels != -100], ids[labels != -100])


def test_validation_masks_do_not_depend_on_corpus_order_or_batch_padding():
    tokenizer = ReactionTokenizer()
    texts = ["CCO>>CC=O", "CCCCN>>CCCC=N"]
    encoded = [tokenizer.encode(text)["input_ids"] for text in texts]
    forward = _fixed_validation(encoded, texts, 55, 0.2)
    reverse = _fixed_validation(encoded[::-1], texts[::-1], 55, 0.2)
    for (ids_a, labels_a), (ids_b, labels_b) in zip(forward, reverse[::-1]):
        assert torch.equal(ids_a, ids_b)
        assert torch.equal(labels_a, labels_b)


def test_exact_train_validation_overlap_is_rejected_before_creating_output(tmp_path):
    train, val = corpus(tmp_path)
    val.write_text("CCO>>CC=O\n")
    with pytest.raises(ValueError, match="exact-text leakage"):
        run(tmp_path, train=train, val=val)
    assert not (tmp_path / "mlm").exists()


def test_explicit_exclusions_filter_both_sets_and_record_counts(tmp_path):
    train, val = corpus(tmp_path)
    val.write_text("CCC=O>>CCCO\nCBr.O>>CO.Br\n")
    exclusions = tmp_path / "held_out.txt"
    exclusions.write_text("CBr.O>>CO.Br\n")
    summary = run(tmp_path, train=train, val=val, exclude_path=exclusions)
    assert summary["train_texts"] == summary["validation_texts"] == 1
    record = json.loads((tmp_path / "mlm/run.json").read_text())
    assert record["train_corpus"]["excluded_texts"] == 1
    assert record["validation_corpus"]["excluded_texts"] == 1
    assert record["exclusion_corpus"]["sha256"]
    assert record["leakage_check"] == "exact_text_only_not_chemical_identity"


def test_excluding_entire_validation_set_does_not_produce_a_checkpoint(tmp_path):
    train, val = corpus(tmp_path)
    exclusions = tmp_path / "held_out.txt"
    exclusions.write_text(val.read_text())
    with pytest.raises(ValueError, match="nonempty after exclusion"):
        run(tmp_path, train=train, val=val, exclude_path=exclusions)
    assert not (tmp_path / "mlm").exists()


def test_csv_column_selection_ignores_labels_but_rejects_test_split(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    for path, rows in [(train, [("CCO>>CC=O", "train"), ("CBr.O>>CO.Br", "train")]),
                       (val, [("CCC=O>>CCCO", "val")])]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["reaction", "split", "log10_kcat"])
            writer.writeheader()
            for text, split in rows:
                writer.writerow(dict(reaction=text, split=split, log10_kcat="not-a-label"))
    assert run(tmp_path, train=train, val=val, text_column="reaction")["status"] == "completed"
    train.write_text(train.read_text().replace(",train,", ",test,"))
    with pytest.raises(ValueError, match="different split"):
        run(tmp_path, train=train, val=val, text_column="reaction",
            output_dir=tmp_path / "second")


@pytest.mark.parametrize("representation,text", [
    ("reaction_smiles", "CCO"),
    ("substrate_smiles", "CCO>>CC=O"),
    ("substrate_smiles", "C{-|=}O"),
    ("sr_smiles", "CCO>>CC=O"),
    ("reaction_smiles", "[CH3:1]O>>[CH2:1]=O"),
])
def test_wrong_representation_or_atom_map_ids_fail_before_training(tmp_path, representation, text):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text(text + "\n")
    val.write_text("N>>C" if representation == "reaction_smiles" else "N")
    with pytest.raises(ValueError):
        run(tmp_path, train=train, val=val, chemical_encoder=representation)
    assert not (tmp_path / "mlm").exists()


def test_sequence_overflow_is_not_silently_truncated(tmp_path):
    train, val = corpus(tmp_path)
    with pytest.raises(ValueError, match="silent truncation"):
        run(tmp_path, train=train, val=val, model_config=replace(tiny_config(), max_length=5))
    assert not (tmp_path / "mlm").exists()


def test_output_cannot_overwrite_input_and_existing_run_requires_explicit_overwrite(tmp_path):
    train, val = corpus(tmp_path)
    with pytest.raises(ValueError, match="must not contain"):
        run(tmp_path, train=train, val=val, output_dir=tmp_path, overwrite=True)
    summary = run(tmp_path, train=train, val=val)
    with pytest.raises(FileExistsError):
        run(tmp_path, train=train, val=val)
    assert (tmp_path / "mlm/best_encoder.pt").exists()
    assert summary["checkpoint_sha256"]


def test_repeated_runs_are_deterministic_for_same_seed(tmp_path):
    train, val = corpus(tmp_path)
    first = run(tmp_path, train=train, val=val, output_dir=tmp_path / "first")
    second = run(tmp_path, train=train, val=val, output_dir=tmp_path / "second")
    assert first["best_val_loss"] == second["best_val_loss"]
    assert (tmp_path / "first/history.jsonl").read_text() == (tmp_path / "second/history.jsonl").read_text()
    a = torch.load(first["checkpoint"], weights_only=True)["encoder"]["encoder_state"]
    b = torch.load(second["checkpoint"], weights_only=True)["encoder"]["encoder_state"]
    for key in a:
        assert torch.equal(a[key], b[key])


@pytest.mark.parametrize("probability", [0, -0.1, 1.1, float("nan"), True])
def test_invalid_mask_probability_rejected(tmp_path, probability):
    with pytest.raises(ValueError, match="mask_probability"):
        run(tmp_path, mask_probability=probability)
