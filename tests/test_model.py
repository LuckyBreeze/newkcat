"""Failure-mode tests, using toy graphs rather than biochemical efficacy claims."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from newkcat.model import GLOBAL_FEATURE_DIM, MODES, ModelConfig, NewKcat
from newkcat.types import GraphBatch


def make_batch():
    generator = torch.Generator().manual_seed(51)
    # Both graphs have three real atoms. Graph 1 contains an isolated atom.
    edges = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    bond_features = torch.rand((3, 3), generator=generator).repeat_interleave(2, dim=0)
    return GraphBatch(
        atom_features=torch.rand((2, 4, 4), generator=generator),
        edge_index=edges,
        edge_features=bond_features,
        distances=torch.tensor([
            [[0, 1, 2, -1], [1, 0, 1, -1], [2, 1, 0, -1], [-1, -1, -1, -1]],
            [[0, 1, -1, -1], [1, 0, -1, -1], [-1, -1, 0, -1], [-1, -1, -1, -1]],
        ]),
        atom_mask=torch.tensor([[True, True, True, False], [True, True, True, False]]),
        center_scores=torch.tensor([[0.6, 0.1, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]]),
        center_available=torch.tensor([True, True]),
        protein=torch.randn((2, 1024), generator=generator),
        molt5=torch.randn((2, 768), generator=generator),
        maccs=torch.randint(0, 2, (2, 167), generator=generator).float(),
        ec2=torch.tensor([0, 1]),
        target=torch.tensor([1.0, -0.5]),
        sample_ids=("synthetic-a", "synthetic-b"),
    )


def make_model(mode="conditioned", **kwargs):
    torch.manual_seed(12)
    return NewKcat(ModelConfig(atom_dim=4, bond_dim=3, mode=mode, hidden_dim=8,
                              num_layers=2, local_dim=4, dropout=0.0, **kwargs))


def permuted(batch):
    # Move padding into the middle; topology, scores and global data are unchanged.
    permutation = torch.tensor([2, 3, 0, 1])
    inverse = permutation.argsort()
    width = batch.atom_features.shape[1]
    edges = batch.edge_index // width * width + inverse[batch.edge_index % width]
    return replace(batch, atom_features=batch.atom_features[:, permutation],
                   atom_mask=batch.atom_mask[:, permutation],
                   center_scores=batch.center_scores[:, permutation], edge_index=edges,
                   distances=batch.distances[:, permutation][:, :, permutation])


def padded(batch, extra=3):
    batch_size, width, feature_dim = batch.atom_features.shape
    larger_width = width + extra
    # NaN padding is deliberate: zero * NaN would be an invalid masking strategy.
    atoms = torch.full((batch_size, larger_width, feature_dim), float("nan"))
    atoms[:, :width] = batch.atom_features
    mask = torch.zeros((batch_size, larger_width), dtype=torch.bool)
    mask[:, :width] = batch.atom_mask
    scores = torch.zeros((batch_size, larger_width))
    scores[:, :width] = batch.center_scores
    distances = torch.full((batch_size, larger_width, larger_width), -1, dtype=torch.long)
    distances[:, :width, :width] = batch.distances
    edges = batch.edge_index // width * larger_width + batch.edge_index % width
    return replace(batch, atom_features=atoms, atom_mask=mask, center_scores=scores,
                   distances=distances, edge_index=edges)


def first_sample(batch):
    width = batch.atom_features.shape[1]
    edge_mask = batch.edge_index[0] < width
    return replace(batch, atom_features=batch.atom_features[:1],
                   atom_mask=batch.atom_mask[:1], distances=batch.distances[:1],
                   center_scores=batch.center_scores[:1],
                   center_available=batch.center_available[:1],
                   protein=batch.protein[:1], molt5=batch.molt5[:1], maccs=batch.maccs[:1],
                   ec2=batch.ec2[:1], target=batch.target[:1],
                   edge_index=batch.edge_index[:, edge_mask],
                   edge_features=batch.edge_features[edge_mask], sample_ids=batch.sample_ids[:1])


@pytest.mark.parametrize("mode", MODES)
def test_prediction_is_invariant_to_atom_permutation_and_undirected_orientation(mode):
    model, batch = make_model(mode).eval(), make_batch()
    with torch.no_grad():
        original = model(batch)
        reordered = model(permuted(batch))
        reversed_edges = model(replace(batch, edge_index=batch.edge_index.flip(0)))
    assert original.shape == (2,)
    torch.testing.assert_close(original, reordered, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original, reversed_edges, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_eval_prediction_is_independent_of_padding_and_other_batch_members(mode):
    model, batch = make_model(mode).eval(), make_batch()
    with torch.no_grad():
        original = model(batch)
        with_padding = model(padded(batch))
        alone = model(first_sample(batch))
    torch.testing.assert_close(original, with_padding, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original[:1], alone, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["center", "shell", "conditioned"])
def test_missing_or_zero_centers_have_zero_branch_after_every_bias(mode):
    model, batch = make_model(mode).eval(), make_batch()
    # This catches accidental bias leakage even when all aggregation inputs are zero.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "local_projection" in name and name.endswith("bias"):
                parameter.fill_(3.0)
    missing = replace(batch, center_available=torch.tensor([False, False]))
    zeros = replace(batch, center_scores=torch.zeros_like(batch.center_scores))
    different_graph = replace(missing, atom_features=missing.atom_features * 100)
    with torch.no_grad():
        prediction, details = model(missing, return_diagnostics=True)
        zero_prediction, zero_details = model(zeros, return_diagnostics=True)
        changed_prediction = model(different_graph)
    assert details["local"].count_nonzero() == 0
    assert zero_details["local"].count_nonzero() == 0
    assert not bool(details["branch_active"].any())
    assert details["slot_weights"].count_nonzero() == 0
    torch.testing.assert_close(prediction, zero_prediction)
    torch.testing.assert_close(prediction, changed_prediction)


@pytest.mark.parametrize("mode", ["global", "learned"])
def test_center_free_controls_cannot_observe_center_scores_or_availability(mode):
    model, batch = make_model(mode).eval(), make_batch()
    missing = replace(batch, center_scores=torch.zeros_like(batch.center_scores),
                      center_available=torch.tensor([False, False]))
    with torch.no_grad():
        prediction, details = model(batch, return_diagnostics=True)
        alternate, alternate_details = model(missing, return_diagnostics=True)
    torch.testing.assert_close(prediction, alternate, atol=0, rtol=0)
    torch.testing.assert_close(details["local"], alternate_details["local"], atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_have_finite_vector_outputs_and_finite_nonzero_backward(mode):
    model, batch = make_model(mode).train(), make_batch()
    prediction = model(batch)
    assert prediction.shape == batch.target.shape == (2,)
    assert bool(torch.isfinite(prediction).all())
    (prediction - batch.target).square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    assert sum(float(grad.abs().sum()) for grad in gradients) > 0
    if mode == "conditioned":
        assert model.position_encoder.weight.grad.abs().sum() > 0
        assert model.enzyme_query.weight.grad.abs().sum() > 0


def test_missing_centers_still_allow_finite_training_backward():
    model, batch = make_model().train(), make_batch()
    batch = replace(batch, center_available=torch.tensor([False, False]))
    prediction = model(batch)
    (prediction - batch.target).square().mean().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())


def test_official_catapro_initialization_preserves_original_predictions_and_zeroes_new_columns():
    # Direct original architecture is independent of NewKcat's baseline forward.
    class OriginalRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.prot_norm = nn.BatchNorm1d(1024)
            self.molt5_norm = nn.BatchNorm1d(768)
            self.decoder = nn.Sequential(nn.Linear(1959, 256), nn.BatchNorm1d(256),
                                         nn.Dropout(0.1), nn.ReLU())
            self.out = nn.Sequential(nn.Linear(256, 1))

        def forward(self, batch):
            features = torch.cat((self.prot_norm(batch.protein),
                                  self.molt5_norm(batch.molt5), batch.maccs), dim=1)
            return self.out(self.decoder(features)).squeeze(-1)

    original, batch = OriginalRegressor(), make_batch()
    original(batch)  # Include non-default BN running statistics in the transfer.
    original.eval()
    for mode in MODES:
        extended = make_model(mode, use_ec2=True, ec_vocab_size=2).eval()
        extended.load_catapro_weights(original.state_dict())
        assert extended.decoder[0].weight[:, GLOBAL_FEATURE_DIM:].count_nonzero() == 0
        with torch.no_grad():
            torch.testing.assert_close(extended(batch), original(batch), atol=1e-6, rtol=1e-6)


def test_checkpoint_validation_is_atomic_and_accepts_dataparallel_prefix():
    model = make_model()
    official = make_model("baseline").state_dict()
    prefixed = {"module." + key: value for key, value in official.items()}
    model.load_catapro_weights(prefixed)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    invalid = {key: value.clone() for key, value in official.items()}
    invalid["decoder.0.weight"] = torch.zeros((256, 1958))
    invalid["prot_norm.running_mean"].fill_(99)
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_catapro_weights(invalid)
    for key, tensor in before.items():
        torch.testing.assert_close(model.state_dict()[key], tensor, atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["nan", "variance", "keys", "dtype"])
def test_corrupt_official_weights_fail_before_nan_predictions(corruption):
    model = make_model()
    official = {key: value.clone() for key, value in make_model("baseline").state_dict().items()}
    if corruption == "nan":
        official["out.0.weight"].fill_(float("nan"))
    elif corruption == "variance":
        official["prot_norm.running_var"][0] = -1
    elif corruption == "keys":
        official[1] = official.pop("out.0.bias")
    else:
        official["out.0.weight"] = official["out.0.weight"].long()
    with pytest.raises(ValueError):
        model.load_catapro_weights(official)


def test_singleton_training_and_broadcastable_targets_fail_with_clear_errors():
    model, batch = make_model(), make_batch()
    with pytest.raises(ValueError, match="BatchNorm requires at least two"):
        model(first_sample(batch))
    with pytest.raises(ValueError, match="broadcasting risk"):
        model(replace(batch, target=batch.target[:, None]))


@pytest.mark.parametrize("problem", ["cross_graph", "padding", "missing_reverse", "features"])
def test_malformed_graph_edges_are_rejected(problem):
    model, batch = make_model().eval(), make_batch()
    edges = batch.edge_index.clone()
    features = batch.edge_features.clone()
    if problem == "cross_graph":
        edges[1, 0] = 4
    elif problem == "padding":
        edges[1, 0] = 3
    elif problem == "missing_reverse":
        edges, features = edges[:, 1:], features[1:]
    else:
        features[0, 0] += 0.5
    with pytest.raises(ValueError):
        model(replace(batch, edge_index=edges, edge_features=features))


def test_real_nan_features_and_out_of_vocabulary_ec_fail_before_prediction():
    batch = make_batch()
    atoms = batch.atom_features.clone()
    atoms[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Real atom features"):
        make_model().eval()(replace(batch, atom_features=atoms))
    with pytest.raises(ValueError, match="EC index outside"):
        make_model(use_ec2=True, ec_vocab_size=1).eval()(batch)


@pytest.mark.parametrize("dropout", ["0.1", [0.1], False, float("nan"), float("inf"), -0.1, 1])
def test_invalid_dropout_is_a_clear_configuration_error(dropout):
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(atom_dim=4, bond_dim=3, dropout=dropout)
