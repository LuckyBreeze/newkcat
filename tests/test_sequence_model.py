"""Sequence-model checks on synthetic inputs; no biochemical performance claims."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from newkcat.sequence.model import (
    BOS_ID, EOS_ID, MASK_ID, PAD_ID, VOCAB_SIZE,
    ReactionEncoder, ReactionMLM, ReactionSequenceModel, SequenceModelConfig,
)


def config(pooling="mean", **kwargs):
    options = dict(d_model=12, nhead=3, num_layers=2, dim_feedforward=24,
                   dropout=0.0, max_length=32, head_dim=16, pooling=pooling)
    options.update(kwargs)
    return SequenceModelConfig(**options)


def batch():
    generator = torch.Generator().manual_seed(19)
    return SimpleNamespace(
        protein=torch.randn(2, 1024, generator=generator),
        input_ids=torch.tensor([[BOS_ID, 38, 39, 45, EOS_ID, PAD_ID],
                                [BOS_ID, 39, 91, 57, 40, EOS_ID]]),
        attention_mask=torch.tensor([[True, True, True, True, True, False],
                                     [True, True, True, True, True, True]]),
        edit_mask=torch.tensor([[False, False, True, True, False, False],
                                [False, True, False, True, False, False]]),
        target=torch.tensor([1.2, -0.8]),
    )


def changed(original, **kwargs):
    return SimpleNamespace(**(vars(original) | kwargs))


def model(pooling="mean", **kwargs):
    torch.manual_seed(31)
    return ReactionSequenceModel(config(pooling, **kwargs))


@pytest.mark.parametrize("pooling", ["mean", "edit_conditioned"])
def test_replacement_model_needs_only_protein_tokens_and_masks(pooling):
    value = batch()
    network = model(pooling)
    prediction = network(value)
    assert prediction.shape == value.target.shape == (2,)
    assert torch.isfinite(prediction).all()
    F.mse_loss(prediction, value.target).backward()
    assert network.encoder.token_embedding.weight.grad.abs().sum() > 0
    assert network.protein_projection[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    if pooling == "edit_conditioned":
        assert network.enzyme_query.weight.grad.abs().sum() > 0
    assert not hasattr(value, "molt5")
    assert not hasattr(value, "maccs")


@pytest.mark.parametrize("pooling", ["mean", "edit_conditioned"])
def test_right_padding_and_batch_composition_do_not_change_eval_outputs(pooling):
    value, network = batch(), model(pooling).eval()
    padded = changed(
        value,
        input_ids=F.pad(value.input_ids, (0, 5), value=PAD_ID),
        attention_mask=F.pad(value.attention_mask, (0, 5), value=False),
        edit_mask=F.pad(value.edit_mask, (0, 5), value=False),
    )
    padded.input_ids[~padded.attention_mask] = 97  # Arbitrary legal IDs on masked positions.
    alone = SimpleNamespace(**{key: tensor[:1] for key, tensor in vars(value).items()})
    reversed_rows = SimpleNamespace(**{key: tensor.flip(0) for key, tensor in vars(value).items()})
    with torch.no_grad():
        reference = network(value)
        torch.testing.assert_close(network(padded), reference, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(network(alone), reference[:1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(network(reversed_rows).flip(0), reference, atol=1e-6, rtol=1e-6)


def test_mean_control_cannot_observe_valid_edit_labels():
    value, network = batch(), model().eval()
    altered = changed(value, edit_mask=torch.zeros_like(value.edit_mask))
    with torch.no_grad():
        torch.testing.assert_close(network(value), network(altered), rtol=0, atol=0)


def test_missing_edits_zero_local_after_biases_but_keep_global_reaction_information():
    value, network = batch(), model("edit_conditioned").eval()
    missing = changed(value, edit_mask=torch.zeros_like(value.edit_mask))
    with torch.no_grad():
        for name, parameter in network.local_projection.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(5.0)
        prediction, details = network(missing, return_diagnostics=True)
    assert torch.isfinite(prediction).all()
    assert details["local"].count_nonzero() == 0
    assert details["edit_weights"].count_nonzero() == 0
    assert not details["edit_active"].any()
    assert details["pooled"].count_nonzero() > 0
    network.train()
    F.mse_loss(network(missing), missing.target).backward()
    assert network.encoder.token_embedding.weight.grad.abs().sum() > 0


def test_local_weights_only_cover_edited_characters_and_ignore_empty_rows():
    value, network = batch(), model("edit_conditioned").eval()
    edits = value.edit_mask.clone()
    edits[0] = False
    with torch.no_grad():
        _, details = network(changed(value, edit_mask=edits), return_diagnostics=True)
    assert details["edit_weights"][~edits].count_nonzero() == 0
    torch.testing.assert_close(details["edit_weights"].sum(1), torch.tensor([0.0, 1.0]))
    assert details["local"][0].count_nonzero() == 0


def test_protein_query_changes_local_readout_for_an_identical_reaction():
    value, network = batch(), model("edit_conditioned").eval()
    value.input_ids[1] = value.input_ids[0]
    value.attention_mask[1] = value.attention_mask[0]
    value.edit_mask[1] = value.edit_mask[0]
    with torch.no_grad():
        _, details = network(value, return_diagnostics=True)
    torch.testing.assert_close(details["pooled"][0], details["pooled"][1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(details["edit_weights"][0], details["edit_weights"][1])


def test_mlm_has_encoder_gradients_and_transfers_without_a_regression_head():
    value = batch()
    pretrainer = ReactionMLM(config())
    ids = value.input_ids.clone()
    labels = torch.full_like(ids, -100)
    labels[:, 2] = ids[:, 2]
    ids[:, 2] = MASK_ID
    logits = pretrainer(ids, value.attention_mask)
    assert logits.shape == (*ids.shape, VOCAB_SIZE)
    assert logits[~value.attention_mask].count_nonzero() == 0
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert pretrainer.encoder.token_embedding.weight.grad.abs().sum() > 0
    assert pretrainer.lm_head.weight.grad.abs().sum() > 0
    transferred = ReactionSequenceModel(config("edit_conditioned", head_dim=8)).eval()
    payload = pretrainer.encoder.export_encoder("fixed-ascii-v1-test")
    transferred.encoder.load_encoder(payload, "fixed-ascii-v1-test")
    pretrainer.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            pretrainer.encoder(value.input_ids, value.attention_mask),
            transferred.encoder(value.input_ids, value.attention_mask), rtol=0, atol=0,
        )
    assert not any("lm_head" in key for key in payload["encoder_state"])


@pytest.mark.parametrize("corruption", ["tokenizer", "config", "nan", "missing", "dtype"])
def test_failed_encoder_load_is_atomic(corruption):
    destination = ReactionEncoder(config())
    source = ReactionEncoder(config())
    before = {key: value.clone() for key, value in destination.state_dict().items()}
    payload = source.export_encoder("fixed-v1")
    if corruption == "tokenizer":
        payload["tokenizer_fingerprint"] = "wrong-vocab"
    elif corruption == "config":
        payload["encoder_config"]["nhead"] = 2
    elif corruption == "nan":
        payload["encoder_state"]["token_embedding.weight"][1, 0] = float("nan")
    elif corruption == "missing":
        payload["encoder_state"].pop("token_embedding.weight")
    else:
        payload["encoder_state"]["token_embedding.weight"] = (
            payload["encoder_state"]["token_embedding.weight"].double()
        )
    with pytest.raises(ValueError):
        destination.load_encoder(payload, "fixed-v1")
    for key, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_exported_encoder_payload_uses_clones_not_shared_storage():
    encoder = ReactionEncoder(config())
    payload = encoder.export_encoder("fixed-v1")
    with torch.no_grad():
        encoder.token_embedding.weight.fill_(42.0)
    assert not torch.equal(payload["encoder_state"]["token_embedding.weight"],
                           encoder.token_embedding.weight)


def test_frozen_encoder_stays_in_eval_and_head_can_train():
    value, network = batch(), model(dropout=0.2)
    network.set_encoder_frozen(True)
    network.train()
    assert network.training and not network.encoder.training
    assert all(not parameter.requires_grad for parameter in network.encoder.parameters())
    F.mse_loss(network(value), value.target).backward()
    assert all(parameter.grad is None for parameter in network.encoder.parameters())
    assert network.regressor[0].weight.grad.abs().sum() > 0
    network.set_encoder_frozen(False)
    assert network.encoder.training
    assert all(parameter.requires_grad for parameter in network.encoder.parameters())


@pytest.mark.parametrize("corruption", [
    "ids_float", "ids_outside", "mask_float", "mask_wrong_shape", "pad_visible",
    "no_characters", "too_long", "protein_nan", "protein_shape", "target_matrix",
    "target_nan", "edit_padding", "edit_bos", "edit_dtype",
])
def test_malformed_model_inputs_fail_explicitly(corruption):
    value, network = batch(), model()
    if corruption == "ids_float":
        value.input_ids = value.input_ids.float()
    elif corruption == "ids_outside":
        value.input_ids[0, 1] = VOCAB_SIZE
    elif corruption == "mask_float":
        value.attention_mask = value.attention_mask.float()
    elif corruption == "mask_wrong_shape":
        value.attention_mask = value.attention_mask[:, :-1]
    elif corruption == "pad_visible":
        value.attention_mask[0, -1] = True
    elif corruption == "no_characters":
        value.attention_mask[0] = False
        value.attention_mask[0, 0] = True
    elif corruption == "too_long":
        value.input_ids = F.pad(value.input_ids, (0, 30), value=PAD_ID)
        value.attention_mask = F.pad(value.attention_mask, (0, 30), value=False)
        value.edit_mask = F.pad(value.edit_mask, (0, 30), value=False)
    elif corruption == "protein_nan":
        value.protein[0, 0] = float("nan")
    elif corruption == "protein_shape":
        value.protein = value.protein[:, :-1]
    elif corruption == "target_matrix":
        value.target = value.target[:, None]
    elif corruption == "target_nan":
        value.target[0] = float("inf")
    elif corruption == "edit_padding":
        value.edit_mask[0, -1] = True
    elif corruption == "edit_bos":
        value.edit_mask[:, 0] = True
    else:
        value.edit_mask = value.edit_mask.float()
    with pytest.raises(ValueError):
        network(value)


@pytest.mark.parametrize("field,value", [
    ("d_model", 7), ("nhead", 0), ("num_layers", True), ("dropout", float("nan")),
    ("dropout", 1.0), ("dropout", True), ("pooling", "attention"),
    ("max_length", 2), ("vocab_size", 100), ("protein_dim", -1),
])
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(config(), **{field: value})


def test_sinusoidal_positions_support_an_odd_hidden_width():
    value = batch()
    encoder = ReactionEncoder(config(d_model=9, nhead=3))
    assert encoder(value.input_ids, value.attention_mask).shape == (2, 6, 9)


def test_encoder_layers_are_initialized_independently():
    encoder = ReactionEncoder(config(num_layers=2))
    first, second = encoder.transformer.layers
    assert not torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight)
