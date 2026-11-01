"""Contracts that prevent truncation, fabricated topology and random backbones."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
import torch

from newkcat.interaction.chemistry import (
    BOND_AROMATIC,
    BOND_OTHER,
    BOND_SELF,
    BOND_SINGLE,
    tokenize_reaction,
)
from newkcat.interaction import encoders
from newkcat.interaction.encoders import (
    ProteinResidueEncoder,
    ReactionTokenEncoder,
    window_slices,
)


VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "C", "O", ">>", "*", "[CH3]", "[OH]", "!"]


def checkpoint(tmp_path, max_positions=512):
    directory = tmp_path / "rxnfp"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "vocab_size": len(VOCAB),
        "hidden_size": 4,
        "max_position_embeddings": max_positions,
    }))
    (directory / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    # Mock loading is intentional; a real, locally saved BertModel is tested below.
    (directory / "pytorch_model.bin").write_bytes(b"mock_checkpoint")
    return directory


class MockEncoder(torch.nn.Module):
    def __init__(self, width=4, vocab_size=len(VOCAB)):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.dropout = torch.nn.Dropout(0.9)
        self.config = SimpleNamespace(hidden_size=width)
        self.width = width
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, **kwargs):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.calls.append(input_ids.detach().cpu().clone())
        result = input_ids.float().unsqueeze(-1).repeat(1, 1, self.width)
        result[:, :, 1] = torch.arange(input_ids.shape[1], device=input_ids.device)
        return SimpleNamespace(last_hidden_state=self.dropout(result))


def mock_rxn_load(monkeypatch, model=None, loading_info=None):
    model = model or MockEncoder()
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return model, (loading_info or {})

    monkeypatch.setattr(encoders, "_transformers", lambda: SimpleNamespace(
        BertModel=SimpleNamespace(from_pretrained=load), __version__="mock-version"
    ))
    return model, calls


def test_token_spans_cover_maps_and_unmatched_characters():
    source = "[CH3:42][OH:99].[*:7]>>CO!\t"
    result = tokenize_reaction(source)
    assert result.tokens[:4] == ("[CH3]", "[OH]", ".", "*")
    assert result.metadata["atom_map_numbers_removed"] == 3
    assert all(":" not in token for token in result.tokens)
    assert result.tokens[-2:] == ("!", "\t")
    assert result.metadata["unmatched_character_count"] == 2
    assert result.spans[0].tolist() == [0, 8]
    assert result.spans[0, 0] == 0 and result.spans[-1, 1] == len(source)
    np.testing.assert_array_equal(result.spans[1:, 0], result.spans[:-1, 1])
    assert "".join(source[a:b] for a, b in result.spans) == source


@pytest.mark.parametrize("source,expected", [
    ("CO>>C", [0, 0, 3, 2]),
    ("C>O>N", [0, 3, 1, 3, 2]),
    ("C.O>>CO", [0, 0, 0, 3, 2, 2]),
])
def test_reaction_roles(source, expected):
    result = tokenize_reaction(source)
    assert result.roles.tolist() == expected
    assert result.metadata["reaction_separator_status"] == "valid"


def test_branched_ring_atoms_align_to_same_smiles_traversal():
    smiles = "CC(C)c1ccccc1"
    result = tokenize_reaction(smiles + ">>CO")
    reactant_positions = np.where(result.atom_mask & (result.roles == 0))[0]
    mol = Chem.MolFromSmiles(smiles)
    assert len(reactant_positions) == mol.GetNumAtoms()
    typed = {(int(a), int(b)): int(t) for (a, b), t in
             zip(result.edge_index.T, result.edge_types)}
    for bond in mol.GetBonds():
        a = reactant_positions[bond.GetBeginAtomIdx()]
        b = reactant_positions[bond.GetEndAtomIdx()]
        expected = BOND_AROMATIC if bond.GetIsAromatic() else BOND_SINGLE
        assert typed[a, b] == typed[b, a] == expected
    assert all(result.roles[a] == result.roles[b] for a, b in result.edge_index.T)


def test_wildcard_and_explicit_hydrogen_are_real_topology_nodes():
    result = tokenize_reaction("[H][C@H](*)O>>CO")
    positions = np.where(result.atom_mask & (result.roles == 0))[0]
    assert len(positions) == 4
    assert np.all(result.topology_mask[positions])
    self_nodes = result.edge_index[0, result.edge_types == BOND_SELF]
    np.testing.assert_array_equal(np.sort(self_nodes), np.flatnonzero(result.topology_mask))
    assert result.topology_mask[result.tokens.index("*")]


def test_dative_bond_arrow_is_not_a_reaction_separator():
    result = tokenize_reaction("N->[Cu+2]>>N.[Cu+2]")
    assert result.metadata["reaction_separator_status"] == "valid"
    assert result.roles.tolist() == [0, 0, 0, 0, 3, 2, 2, 2]
    assert result.edge_types.tolist().count(BOND_OTHER) == 2
    assert result.metadata["components"][0]["topology_status"] == "available"


@pytest.mark.parametrize("component", ["[NH3]->[Cu+2]", "[Cu+2]<-[NH3]"])
def test_dative_bonds_on_both_sides_preserve_roles_and_topology(component):
    result = tokenize_reaction(component + ">>" + component)
    assert result.metadata["reaction_separator_count"] == 2
    assert result.metadata["reaction_separator_status"] == "valid"
    assert result.roles.tolist() == [0, 0, 0, 0, 3, 2, 2, 2, 2]
    assert result.topology_mask.sum() == result.atom_mask.sum() == 4
    assert result.edge_types.tolist().count(BOND_OTHER) == 4
    assert len(result.metadata["components"]) == 2
    if "<-" in component:
        assert result.metadata["unmatched_character_count"] == 2
        assert result.tokens.count("<") == 2


def test_invalid_component_only_loses_its_own_topology():
    result = tokenize_reaction("CCO.C!C>>CO")
    records = result.metadata["components"]
    assert [record["topology_status"] for record in records] == [
        "available", "unparseable", "available"
    ]
    invalid = records[1]
    left, right = invalid["token_start"], invalid["token_stop"]
    assert result.atom_mask[left:right].sum() == 2
    assert not result.topology_mask[left:right].any()
    assert not np.isin(result.edge_index, np.arange(left, right)).any()
    assert "!" in result.tokens
    assert all(result.topology_mask[node] for node in result.edge_index.ravel())


def test_rdkit_must_not_treat_trailing_text_as_a_name():
    result = tokenize_reaction("CC name>>CO")
    component = result.metadata["components"][0]
    assert component["topology_status"] == "unparseable"
    assert not result.topology_mask[result.roles == 0].any()
    assert result.metadata["unmatched_character_count"] > 0


def test_window_slices_cover_boundaries():
    assert list(window_slices(510, 510, 128)) == [(0, 510)]
    assert list(window_slices(511, 510, 128)) == [(0, 510), (382, 511)]
    assert list(window_slices(1, 510, 128)) == [(0, 1)]
    for invalid in [(0, 10, 2), (4, 0, 0), (4, 10, 10), (4, 10, -1)]:
        with pytest.raises(ValueError):
            list(window_slices(*invalid))


def test_reaction_loader_requires_actual_local_checkpoint_and_preserves_source(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    model, calls = mock_rxn_load(monkeypatch)
    encoder = ReactionTokenEncoder(path)
    assert calls == [(str(path.resolve()), {
        "local_files_only": True, "add_pooling_layer": False, "output_loading_info": True
    })]
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not model.training
    assert encoder.metadata["checkpoint_files_sha256"]["pytorch_model.bin"] == hashlib.sha256(
        b"mock_checkpoint"
    ).hexdigest()
    with pytest.raises(ValueError, match="local directory"):
        ReactionTokenEncoder("missing-hf-model-or-directory")


def test_reaction_loader_rejects_randomly_initialised_missing_backbone(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    mock_rxn_load(monkeypatch, loading_info={"missing_keys": ["encoder.layer.0.attention.self.query.weight"]})
    with pytest.raises(ValueError, match="complete pretrained backbone"):
        ReactionTokenEncoder(path)


def test_reaction_long_windows_retain_all_tokens_and_average_overlap(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    model, _ = mock_rxn_load(monkeypatch)
    encoder = ReactionTokenEncoder(path, overlap=128)
    text = "C" * 1100 + ">>" + "O" * 10
    result = encoder.encode(text)
    states = result["reaction_states"]
    assert states.shape == (1111, 4)
    assert all(call.shape[1] <= 512 for call in model.calls)
    assert all(call[0, 0] == 2 and call[0, -1] == 3 for call in model.calls)
    assert result["metadata"]["window_count"] == 3
    np.testing.assert_array_equal(states[:, 0], [4] * 1100 + [6] + [5] * 10)
    sums = np.zeros(len(states))
    counts = np.zeros(len(states))
    for start, stop in result["metadata"]["window_specs"]:
        sums[start:stop] += np.arange(1, stop - start + 1)
        counts[start:stop] += 1
    np.testing.assert_allclose(states[:, 1], sums / counts)
    assert result["metadata"]["raw_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_reaction_map_numbers_do_not_reach_model_and_dropout_stays_off(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    model, _ = mock_rxn_load(monkeypatch)
    encoder = ReactionTokenEncoder(path)
    first = encoder.encode("[CH3:2][OH:3]>>C*")
    model.train()  # A cache-populating call restores eval, even after this mistake.
    second = encoder.encode("[CH3:923][OH:111]>>C*")
    np.testing.assert_array_equal(first["reaction_states"], second["reaction_states"])
    assert second["metadata"]["oov_token_count"] == 0
    assert not model.training
    assert first["metadata"]["token_spans"] != second["metadata"]["token_spans"]


def test_unrecognised_character_gets_unk_state_instead_of_disappearing(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    mock_rxn_load(monkeypatch)
    result = ReactionTokenEncoder(path).encode("C!O>>C")
    assert result["reaction_states"].shape[0] == 5
    assert result["reaction_states"][1, 0] == VOCAB.index("[UNK]")
    assert result["metadata"]["oov_token_indices"] == [1]


class MockProteinTokenizer:
    name_or_path = "mock-ProtT5"
    unk_token_id = 1

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 128

    def __call__(self, sequence, **kwargs):
        self.calls.append((sequence, kwargs))
        ids = [ord(token) for token in sequence.split()] + [0]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "special_tokens_mask": torch.tensor([[0] * (len(ids) - 1) + [1]]),
        }


def mock_protein_load(monkeypatch, width=1024):
    model = MockEncoder(width=width, vocab_size=128)
    model.config = SimpleNamespace(
        d_model=width, _commit_hash="resolved-mock-sha", to_dict=lambda: {"d_model": width}
    )
    tokenizer = MockProteinTokenizer()
    calls = []

    def load_model(source, **kwargs):
        calls.append((source, kwargs))
        return model, {}

    monkeypatch.setattr(encoders, "_transformers", lambda: SimpleNamespace(
        T5EncoderModel=SimpleNamespace(from_pretrained=load_model),
        T5Tokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        __version__="mock-version",
    ))
    return model, tokenizer, calls


def test_protein_long_windows_preserve_every_residue_and_normalise_uzob(monkeypatch):
    model, tokenizer, calls = mock_protein_load(monkeypatch)
    encoder = ProteinResidueEncoder("mock-ProtT5", window_size=1000, overlap=128)
    sequence = "UZOB" + "ACDEFGHIKLMNPQRSTVWY" * 110
    result = encoder.encode(sequence)
    states = result["residue_states"]
    assert states.shape == (len(sequence), 1024)
    expected = "XXXX" + sequence[4:]
    np.testing.assert_array_equal(states[:, 0], [ord(residue) for residue in expected])
    assert result["metadata"]["uzob_to_x_positions"] == [0, 1, 2, 3]
    assert all(call.shape[1] <= 1001 for call in model.calls)
    assert all(kwargs["truncation"] is False for _, kwargs in tokenizer.calls)
    assert calls[0][1]["local_files_only"] is True
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert result["metadata"]["resolved_revision"] == "resolved-mock-sha"
    model.train()
    repeat = encoder.encode(sequence)
    np.testing.assert_array_equal(states, repeat["residue_states"])
    assert not model.training


def test_protein_rejects_wrong_width_and_nonsequence_inputs(monkeypatch):
    mock_protein_load(monkeypatch, width=128)
    with pytest.raises(ValueError, match="1024"):
        ProteinResidueEncoder("not-ProtT5")
    mock_protein_load(monkeypatch)
    encoder = ProteinResidueEncoder("mock-ProtT5")
    for bad in ("", "A C", ">seq\nAC", "AC*", "ACé"):
        with pytest.raises(ValueError):
            encoder.encode(bad)


def test_real_saved_bert_checkpoint_loads_without_rxnfp_dependency(tmp_path):
    """No internet, downloaded weights, or large protein model is required."""
    transformers = pytest.importorskip("transformers")
    path = tmp_path / "tiny_saved_bert"
    config = transformers.BertConfig(
        vocab_size=len(VOCAB), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
    )
    original = transformers.BertModel(config, add_pooling_layer=False).eval()
    original.save_pretrained(path)
    (path / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    encoder = ReactionTokenEncoder(path, overlap=4)
    text = "CCO>>C*"
    result = encoder.encode(text)
    ids = torch.tensor([[2, 4, 4, 5, 6, 4, 7, 3]])
    with torch.no_grad():
        expected = original(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    np.testing.assert_allclose(result["reaction_states"], expected[0, 1:-1].numpy(), atol=1e-6)
    assert result["metadata"]["checkpoint_files_sha256"]["model.safetensors"]
