"""Numerical invariants and controlled-ablation contracts for reaction readouts."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from newkcat.readout.model import ReactionReadoutModel, ReadoutModelConfig
from newkcat.sequence.data import SequenceBatch
from newkcat.sequence.model import ReactionSequenceModel
from newkcat.sequence.tokenizer import ReactionTokenizer


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**changes):
    base = ReadoutModelConfig(
        protein_dim=12, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, head_dim=8, chemical_dim=8,
        max_length=20, dropout=0.0, num_queries=3,
    )
    return replace(base, **changes)


def batch(protein_dim=12, extra_padding=0):
    ids = torch.tensor([[1, 7, 8, 9, 2, 0], [1, 7, 9, 2, 0, 0]], dtype=torch.long)
    attention = ids != 0
    if extra_padding:
        # Arbitrary valid IDs at masked positions must never influence a result.
        ids = torch.cat((ids, torch.full((2, extra_padding), 17)), dim=1)
        attention = torch.cat((attention, torch.zeros((2, extra_padding), dtype=torch.bool)), dim=1)
    return SequenceBatch(
        protein=torch.arange(2 * protein_dim, dtype=torch.float32).reshape(2, protein_dim)
        / protein_dim,
        input_ids=ids, attention_mask=attention, edit_mask=torch.zeros_like(attention),
        target=torch.tensor([1.0, -1.0]), sample_ids=("a", "b"), statuses=("ok", "ok"),
    )


@pytest.mark.parametrize("training", [False, True])
def test_default_path_is_exact_original_r(training):
    cfg = config(dropout=0.15)
    torch.manual_seed(668)
    original = ReactionSequenceModel(cfg.to_sequence_config())
    torch.manual_seed(668)
    ablation = ReactionReadoutModel(cfg)
    assert original.state_dict().keys() == ablation.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, ablation.state_dict()[key]), key
    original.train(training)
    ablation.train(training)
    torch.manual_seed(42)
    expected = original(batch())
    torch.manual_seed(42)
    actual = ablation(batch())
    assert torch.equal(expected, actual)
    ablation.load_state_dict(original.state_dict(), strict=True)


@pytest.mark.parametrize("protein_readout", ["projected", "full"])
@pytest.mark.parametrize("chemical_dim", [8, None])
@pytest.mark.parametrize("readout", ["mean", "mean_max", "queries", "flatten"])
def test_modes_forward_backward_and_dimensions(protein_readout, chemical_dim, readout):
    cfg = config(protein_readout=protein_readout, chemical_dim=chemical_dim, readout=readout)
    model = ReactionReadoutModel(cfg)
    prediction, diagnostics = model(batch(), return_diagnostics=True)
    assert prediction.shape == (2,)
    assert diagnostics["protein"].shape == (2, cfg.protein_width)
    assert diagnostics["pooled"].shape == (2, cfg.readout_width)
    assert diagnostics["chemical"].shape == (2, cfg.chemical_width)
    assert model.regressor[0].in_features == cfg.protein_width + cfg.chemical_width
    if protein_readout == "full":
        assert isinstance(model.protein_projection, nn.Identity)
        assert torch.equal(diagnostics["protein"], model.protein_norm(batch().protein))
    if chemical_dim is None:
        assert isinstance(model.chemical_projection, nn.Identity)
        assert torch.equal(diagnostics["chemical"], diagnostics["pooled"])
    (prediction - batch().target).square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize("readout", ["mean", "mean_max", "queries", "flatten"])
@pytest.mark.parametrize("chemical_dim", [8, None])
def test_right_padding_cannot_change_prediction(readout, chemical_dim):
    model = ReactionReadoutModel(config(readout=readout, chemical_dim=chemical_dim)).eval()
    with torch.no_grad():
        a, da = model(batch(), return_diagnostics=True)
        b, db = model(batch(extra_padding=7), return_diagnostics=True)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(da["pooled"], db["pooled"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("condition_queries", [False, True])
@pytest.mark.parametrize("fusion", ["concat", "transformer"])
def test_queries_mask_and_fusion_keep_direct_full_protein(condition_queries, fusion):
    cfg = config(readout="queries", protein_readout="full", chemical_dim=None,
                 condition_queries=condition_queries, fusion=fusion)
    model = ReactionReadoutModel(cfg).eval()
    pred, data = model(batch(), return_diagnostics=True)
    weights = data["query_weights"]
    assert weights.shape == (2, 3, 6)
    assert not bool(weights.masked_select(~data["chemical_mask"][:, None, :]).any())
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones((2, 3)))
    assert torch.equal(data["protein"], model.protein_norm(batch().protein))
    if fusion == "transformer":
        assert data["fusion_tokens"].shape == (2, 4, 8)
    pred.square().mean().backward()
    assert model.readout_queries.grad is not None
    if condition_queries:
        assert model.enzyme_query.weight.grad is not None
    if fusion == "transformer":
        assert model.fusion_protein.weight.grad is not None
    with torch.no_grad():
        other = model(batch(extra_padding=5))
    torch.testing.assert_close(pred.detach(), other, atol=1e-6, rtol=1e-6)


def test_unconditioned_queries_ignore_protein_but_conditioned_queries_do_not():
    first = batch()
    second = replace(first, protein=first.protein.flip(dims=(1,)))
    for conditioned in (False, True):
        model = ReactionReadoutModel(config(
            readout="queries", condition_queries=conditioned, protein_readout="full",
        )).eval()
        _, a = model(first, return_diagnostics=True)
        _, b = model(second, return_diagnostics=True)
        if conditioned:
            assert not torch.allclose(a["query_weights"], b["query_weights"])
        else:
            assert torch.equal(a["query_weights"], b["query_weights"])


def test_full_protein_ablation_is_smaller_than_projected_default():
    projected = ReactionReadoutModel(ReadoutModelConfig())
    full = ReactionReadoutModel(ReadoutModelConfig(protein_readout="full"))
    assert sum(p.numel() for p in projected.parameters()) == 1_037_697
    assert sum(p.numel() for p in full.parameters()) == 971_393


def test_flatten_budget_counts_actual_reducer_and_rejects_before_allocation(monkeypatch):
    projected = config(readout="flatten")
    direct = config(readout="flatten", protein_readout="full", chemical_dim=None)
    assert projected.flatten_dense_parameters() == 20 * 8 * 8 + 8
    assert direct.flatten_dense_parameters() == (20 * 8 + 12) * 8 + 8
    with pytest.raises(ValueError, match="exceeding"):
        replace(direct, max_flatten_parameters=direct.flatten_dense_parameters() - 1)
    # Raising a bounded, explicit budget admits exactly the requested layer.
    allowed = replace(direct, max_flatten_parameters=direct.flatten_dense_parameters())
    model = ReactionReadoutModel(allowed)
    layer = model.regressor[0]
    assert layer.weight.numel() + layer.bias.numel() == allowed.max_flatten_parameters
    with pytest.raises(ValueError, match="Flatten"):
        ReadoutModelConfig(readout="flatten")
    # Even if dataclass immutability is bypassed, do not allocate an unsafe encoder.
    object.__setattr__(allowed, "max_flatten_parameters", 1)
    monkeypatch.setattr("newkcat.readout.model.ReactionEncoder",
                        lambda _: pytest.fail("allocated encoder before budget gate"))
    with pytest.raises(ValueError, match="allocation"):
        ReactionReadoutModel(allowed)


@pytest.mark.parametrize("change", [
    {"protein_readout": "raw"}, {"readout": "unknown"}, {"chemical_dim": 0},
    {"chemical_dim": True}, {"chemical_dim": float("inf")}, {"num_queries": 0},
    {"condition_queries": 1}, {"condition_queries": True}, {"fusion": "attention"},
    {"fusion": "transformer"}, {"fusion_num_layers": 0}, {"max_flatten_parameters": 0},
    {"max_flatten_parameters": float("inf")}, {"max_flatten_parameters": True},
    {"d_model": 7}, {"max_length": 2}, {"dropout": float("nan")},
])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        config(**change)


@pytest.mark.parametrize("corruption", ["protein_shape", "protein_nan", "protein_dtype",
                                        "target_shape", "target_inf", "edit_special",
                                        "no_chemical", "too_long"])
def test_invalid_batch_is_rejected(corruption):
    model = ReactionReadoutModel(config())
    item = batch()
    if corruption == "protein_shape":
        item = replace(item, protein=item.protein[:, :-1])
    elif corruption == "protein_nan":
        item.protein[0, 0] = float("nan")
    elif corruption == "protein_dtype":
        item = replace(item, protein=item.protein.double())
    elif corruption == "target_shape":
        item = replace(item, target=item.target[:, None])
    elif corruption == "target_inf":
        item.target[0] = float("inf")
    elif corruption == "edit_special":
        item.edit_mask[0, 0] = True
    elif corruption == "no_chemical":
        item = replace(item, input_ids=torch.tensor([[1, 2], [1, 2]]),
                       attention_mask=torch.ones((2, 2), dtype=torch.bool),
                       edit_mask=torch.zeros((2, 2), dtype=torch.bool))
    else:
        item = batch(extra_padding=20)
    with pytest.raises(ValueError):
        model(item)


def test_frozen_encoder_stays_deterministic_and_exports_original_contract():
    cfg = config(dropout=0.4, protein_readout="full", chemical_dim=None, readout="queries")
    model = ReactionReadoutModel(cfg)
    assert cfg.encoder_config() == cfg.to_sequence_config().encoder_config()
    assert cfg.encoder_fingerprint() == cfg.to_sequence_config().encoder_fingerprint()
    payload = model.encoder.export_encoder(ReactionTokenizer.fingerprint)
    clone = ReactionReadoutModel(cfg)
    clone.encoder.load_encoder(payload, ReactionTokenizer.fingerprint)
    for key, value in model.encoder.state_dict().items():
        assert torch.equal(value, clone.encoder.state_dict()[key])
    model.set_encoder_frozen(True)
    model.train()
    assert model.training and not model.encoder.training
    assert not any(p.requires_grad for p in model.encoder.parameters())
    item = batch()
    a = model.encoder(item.input_ids, item.attention_mask)
    b = model.encoder(item.input_ids, item.attention_mask)
    assert torch.equal(a, b)
    model(item).square().mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.regressor[0].weight.grad is not None
    model.set_encoder_frozen(False)
    assert model.encoder.training
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_batch_size_one_is_supported_without_batchnorm():
    item = batch()
    single = replace(item, protein=item.protein[:1], input_ids=item.input_ids[:1],
                     attention_mask=item.attention_mask[:1], edit_mask=item.edit_mask[:1],
                     target=item.target[:1], sample_ids=("a",), statuses=("ok",))
    model = ReactionReadoutModel(config(protein_readout="full", chemical_dim=None,
                                       readout="queries", fusion="transformer"))
    assert not any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                   for layer in model.modules())
    prediction = model(single)
    assert prediction.shape == (1,)
    prediction.square().mean().backward()
