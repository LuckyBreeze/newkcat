"""Functional checks of real token/residue interaction and safe missing topology."""

from dataclasses import replace

import pytest
import torch

from newkcat.interaction.model import (
    InteractionModelConfig, MaskedMultiheadAttention, ReactionInteractionModel,
)


@pytest.fixture
def tiny_config():
    return InteractionModelConfig(
        protein_dim=12, reaction_dim=10, d_model=8, nhead=2, num_layers=2,
        dim_feedforward=16, head_hidden_dim=16, head_bottleneck_dim=8, dropout=0.0,
    )


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(417)
    reaction_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]],
                                 dtype=torch.bool)
    protein_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    atom_mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 0]],
                            dtype=torch.bool)
    topology_mask = atom_mask.clone()
    topology_mask[0, 3] = False
    bonds = torch.zeros(2, 7, 7, dtype=torch.long)
    for row, left, right, kind in ((0, 0, 1, 2), (0, 5, 6, 3), (1, 0, 1, 2), (1, 3, 4, 5)):
        bonds[row, left, right] = bonds[row, right, left] = kind
    return {
        "protein_states": torch.randn(2, 5, 12, generator=generator),
        "protein_mask": protein_mask,
        "protein_global": torch.randn(2, 12, generator=generator),
        "reaction_states": torch.randn(2, 7, 10, generator=generator),
        "reaction_mask": reaction_mask,
        "roles": torch.tensor([[0, 0, 3, 1, 3, 2, 2], [0, 0, 3, 2, 2, 3, 3]]),
        "atom_mask": atom_mask,
        "topology_mask": topology_mask,
        "bond_types": bonds,
        "targets": torch.tensor([1.2, -0.8]),
    }


def copy_batch(batch):
    return {key: value.clone() for key, value in batch.items()}


def test_default_dimensions_and_meta_construction():
    config = InteractionModelConfig()
    assert config.head_input_dim == 2304
    with torch.device("meta"):
        model = ReactionInteractionModel(config)
    assert model.regressor[0].in_features == 2304
    assert model.regressor[0].out_features == 512
    assert model.regressor[3].out_features == 128


def test_forward_has_full_global_path_and_named_role_summaries(tiny_config, batch):
    model = ReactionInteractionModel(tiny_config).eval()
    prediction, detail = model(batch, return_diagnostics=True)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    assert detail["reaction_summaries"].shape == (2, 4, 8)
    assert detail["protein_summary"].shape == (2, 8)
    torch.testing.assert_close(detail["head_input"][:, :12], batch["protein_global"],
                               rtol=0, atol=0)
    assert detail["head_input"].shape == (2, 52)
    assert not torch.count_nonzero(detail["reaction_summaries"][1, 3])
    assert not torch.count_nonzero(detail["reaction_pool_weights"][1, 3])


def test_padding_and_padded_values_do_not_change_predictions(tiny_config, batch):
    torch.manual_seed(19)
    model = ReactionInteractionModel(tiny_config).eval()
    with torch.no_grad():
        expected = model(batch)
    padded = copy_batch(batch)
    padded["reaction_states"] = torch.cat((batch["reaction_states"], torch.randn(2, 3, 10) * 100), 1)
    padded["protein_states"] = torch.cat((batch["protein_states"], torch.randn(2, 4, 12) * 100), 1)
    for name in ("reaction_mask", "atom_mask", "topology_mask"):
        padded[name] = torch.cat((batch[name], torch.zeros(2, 3, dtype=torch.bool)), 1)
    padded["protein_mask"] = torch.cat((batch["protein_mask"], torch.zeros(2, 4, dtype=torch.bool)), 1)
    padded["roles"] = torch.cat((batch["roles"], torch.full((2, 3), 3, dtype=torch.long)), 1)
    padded["bond_types"] = torch.zeros(2, 10, 10, dtype=torch.long)
    padded["bond_types"][:, :7, :7] = batch["bond_types"]
    # Existing padded entries need not have zero cached values either.
    padded["reaction_states"][~padded["reaction_mask"]] = -31
    padded["protein_states"][~padded["protein_mask"]] = 43
    with torch.no_grad():
        actual, detail = model(padded, return_diagnostics=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not torch.count_nonzero(detail["reaction_states"][~padded["reaction_mask"]])
    assert not torch.count_nonzero(detail["protein_states"][~padded["protein_mask"]])


def test_bidirectional_attention_uses_many_valid_keys(tiny_config, batch):
    model = ReactionInteractionModel(tiny_config).eval()
    _, detail = model(batch, return_diagnostics=True)
    for block in detail["blocks"]:
        chemical_to_protein = block["reaction_to_protein_attention"]
        protein_to_chemical = block["protein_to_reaction_attention"]
        assert chemical_to_protein.shape == (2, 2, 7, 5)
        assert protein_to_chemical.shape == (2, 2, 5, 7)
        assert torch.all((chemical_to_protein[0, :, 0] > 0).sum(-1) == 5)
        assert torch.all((protein_to_chemical[0, :, 0] > 0).sum(-1) == 7)
        torch.testing.assert_close(chemical_to_protein[0].sum(-1), torch.ones(2, 7))
        torch.testing.assert_close(protein_to_chemical[0].sum(-1), torch.ones(2, 5))
        assert not torch.count_nonzero(chemical_to_protein[1, :, :, 3:])
        assert not torch.count_nonzero(protein_to_chemical[1, :, :, 5:])


def test_each_modality_changes_the_other_before_pooling(tiny_config, batch):
    torch.manual_seed(71)
    model = ReactionInteractionModel(tiny_config).eval()
    changed_protein, changed_reaction = copy_batch(batch), copy_batch(batch)
    changed_protein["protein_states"][:, 0, 0] += 3.0
    changed_reaction["reaction_states"][:, 0, 0] += 3.0
    with torch.no_grad():
        _, initial = model(batch, True)
        _, from_protein = model(changed_protein, True)
        _, from_reaction = model(changed_reaction, True)
    assert not torch.allclose(initial["reaction_states"], from_protein["reaction_states"])
    assert not torch.allclose(initial["protein_states"], from_reaction["protein_states"])


def test_cross_attention_ablation_removes_early_cross_modal_dependency(tiny_config, batch):
    model = ReactionInteractionModel(replace(tiny_config, use_cross_attention=False)).eval()
    changed = copy_batch(batch)
    changed["protein_states"][:, 0, 0] += 3
    with torch.no_grad():
        _, initial = model(batch, True)
        _, other = model(changed, True)
    torch.testing.assert_close(initial["reaction_states"], other["reaction_states"], rtol=0, atol=0)
    assert all(block["protein_to_reaction_attention"] is None for block in initial["blocks"])


def test_local_heads_obey_bonds_and_reliable_atoms(tiny_config, batch):
    model = ReactionInteractionModel(tiny_config).eval()
    _, details = model(batch, True)
    allowed = batch["bond_types"].gt(0) | torch.eye(7, dtype=torch.bool)[None]
    reliable = batch["topology_mask"]
    allowed = allowed & reliable[:, :, None] & reliable[:, None, :]
    for block in details["blocks"]:
        weights = block["local_attention"]
        assert not torch.count_nonzero(weights.masked_select(~allowed[:, None]))
        assert not torch.count_nonzero(block["local_update"][~reliable])
        # Atom 0 has exactly itself and atom 1 as local keys.
        assert torch.all((weights[0, :, 0] > 0).sum(-1) == 2)


def test_all_missing_topology_keeps_global_token_path_finite(tiny_config, batch):
    model = ReactionInteractionModel(tiny_config).eval()
    missing = copy_batch(batch)
    missing["topology_mask"].zero_()
    missing["bond_types"].zero_()
    prediction, details = model(missing, True)
    assert torch.isfinite(prediction).all()
    for block in details["blocks"]:
        assert not torch.count_nonzero(block["local_update"])
        assert not torch.count_nonzero(block["local_attention"])
    assert torch.count_nonzero(details["reaction_states"][missing["reaction_mask"]])
    # A non-atomic separator still participates in the global chemical path.
    changed = copy_batch(missing)
    changed["reaction_states"][0, 2, 0] += 4
    assert not torch.allclose(model(changed), prediction)


def test_no_topology_ablation_has_no_local_branch(tiny_config, batch):
    model = ReactionInteractionModel(replace(tiny_config, use_topology=False)).eval()
    _, details = model(batch, True)
    assert not hasattr(model.blocks[0], "local_attention")
    assert all(block["local_attention"] is None for block in details["blocks"])


def test_gradients_reach_two_cross_directions_and_intact_global_cache(tiny_config, batch):
    torch.manual_seed(41)
    model = ReactionInteractionModel(tiny_config)
    batch["protein_global"].requires_grad_()
    prediction = model(batch)
    (prediction - batch["targets"]).square().mean().backward()
    assert torch.isfinite(batch["protein_global"].grad).all()
    assert torch.count_nonzero(batch["protein_global"].grad)
    for block in model.blocks:
        for attention in (block.reaction_to_protein, block.protein_to_reaction):
            for projection in (attention.query, attention.key, attention.value):
                gradient = projection.weight.grad
                assert gradient is not None and torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient)
        assert torch.count_nonzero(block.local_attention.value.weight.grad)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_all_masked_attention_does_not_reintroduce_output_bias():
    attention = MaskedMultiheadAttention(8, 2, dropout=0)
    with torch.no_grad():
        attention.output.bias.fill_(4)
    states = torch.randn(1, 3, 8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    output, weights = attention(states, states, valid, valid,
                                pair_mask=torch.zeros(1, 3, 3, dtype=torch.bool),
                                return_weights=True)
    assert not torch.count_nonzero(output)
    assert not torch.count_nonzero(weights)
    output.sum().backward()
    assert torch.isfinite(attention.output.weight.grad).all()


def test_single_sample_training_does_not_require_batchnorm(tiny_config, batch):
    model = ReactionInteractionModel(tiny_config).train()
    singleton = {name: tensor[:1] for name, tensor in batch.items()}
    prediction = model(singleton)
    assert prediction.shape == (1,)
    prediction.square().mean().backward()


@pytest.mark.parametrize("changes", [
    {"d_model": 7, "nhead": 2}, {"num_layers": 0}, {"protein_dim": True},
    {"dropout": float("nan")}, {"residual_scale_init": 0}, {"use_topology": 1},
])
def test_config_rejects_invalid_values(tiny_config, changes):
    with pytest.raises(ValueError):
        replace(tiny_config, **changes)


@pytest.mark.parametrize("key", ["reaction_states", "protein_states", "protein_global"])
def test_nonfinite_caches_fail_clearly(tiny_config, batch, key):
    model = ReactionInteractionModel(tiny_config)
    batch[key].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model(batch)


def test_topology_mask_cannot_label_nonatomic_tokens(tiny_config, batch):
    model = ReactionInteractionModel(tiny_config)
    batch["topology_mask"][0, 2] = True
    with pytest.raises(ValueError, match="subset"):
        model(batch)
