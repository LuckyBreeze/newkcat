"""Adversarial examples for the center-relative representation."""

import pytest
import torch

from newkcat.position import center_relative_profiles, distance_shell_pool


def test_five_bins_keep_a_distant_atom_distinct_from_an_unconnected_atom():
    # 0--1--2--3 plus isolated atom 4. The final slot is padding.
    distances = torch.full((1, 6, 6), -1, dtype=torch.long)
    distances[0, :4, :4] = torch.abs(torch.arange(4)[:, None] - torch.arange(4))
    distances[0, 4, 4] = 0
    mask = torch.tensor([[True, True, True, True, True, False]])
    scores = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    result = center_relative_profiles(distances, scores, mask)
    torch.testing.assert_close(result.profiles[0, :5], torch.eye(5))
    assert result.profiles[0, 5].count_nonzero() == 0
    assert result.active.tolist() == [True]


def test_multi_center_profiles_are_distributions_not_nearest_center_distance():
    distances = torch.abs(torch.arange(3)[:, None] - torch.arange(3))[None]
    mask = torch.ones((1, 3), dtype=torch.bool)
    scores = torch.tensor([[0.1, 0, 0.1]])
    result = center_relative_profiles(distances, scores, mask)
    expected = torch.tensor([[[0.5, 0, 0.5, 0, 0], [0, 1, 0, 0, 0],
                              [0.5, 0, 0.5, 0, 0]]])
    torch.testing.assert_close(result.profiles, expected)
    doubled = center_relative_profiles(distances, scores * 2, mask)
    torch.testing.assert_close(result.profiles, doubled.profiles)
    torch.testing.assert_close(doubled.mass, result.mass * 2)
    assert not torch.equal(result.statistics, doubled.statistics)


def test_missing_and_zero_centers_deactivate_all_channels_even_for_nonzero_scores():
    distances = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
    scores = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    result = center_relative_profiles(distances, scores, mask, torch.tensor([False, True]))
    assert not bool(result.active.any())
    assert result.profiles.count_nonzero() == 0
    assert result.statistics.count_nonzero() == 0


def test_shell_pool_does_not_include_nan_padding_or_invent_empty_slots():
    states = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [float("nan"), float("nan")]]])
    profiles = torch.tensor([[[1.0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0],
                              [0.0, 0, 0, 0, 0]]])
    mask = torch.tensor([[True, True, False]])
    tokens, masses, valid = distance_shell_pool(states, profiles, mask)
    torch.testing.assert_close(tokens[0, 0], torch.tensor([8 / 3, 16 / 3]))
    torch.testing.assert_close(tokens[0, 1], torch.tensor([4.0, 8.0]))
    assert tokens[0, 2:].count_nonzero() == 0
    torch.testing.assert_close(masses, torch.tensor([[1.5, 0.5, 0, 0, 0]]))
    assert valid.tolist() == [[True, True, False, False, False]]


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -0.1, 1.1])
def test_bad_center_scores_fail_loudly(bad_score):
    with pytest.raises(ValueError, match="center_scores"):
        center_relative_profiles(torch.zeros((1, 1, 1), dtype=torch.long),
                                 torch.tensor([[bad_score]]), torch.tensor([[True]]))


@pytest.mark.parametrize("problem", ["padding", "asymmetric", "diagonal", "zero_distance"])
def test_invalid_distance_or_padding_inputs_are_rejected(problem):
    distances = torch.tensor([[[0, 1, -1], [1, 0, -1], [-1, -1, -1]]])
    scores = torch.tensor([[0.2, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False]])
    if problem == "padding":
        scores[0, 2] = 0.1
    elif problem == "asymmetric":
        distances[0, 1, 0] = 2
    elif problem == "diagonal":
        distances[0, 0, 0] = 1
    else:
        distances[0, 0, 1] = distances[0, 1, 0] = 0
    with pytest.raises(ValueError):
        center_relative_profiles(distances, scores, mask)


def test_position_computation_supports_finite_score_gradients():
    distances = torch.tensor([[[0, 1, 2], [1, 0, 1], [2, 1, 0]]])
    scores = torch.tensor([[0.2, 0.3, 0.0]], requires_grad=True)
    result = center_relative_profiles(distances, scores, torch.ones((1, 3), dtype=torch.bool))
    (result.profiles.square().sum() + result.statistics.square().sum()).backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())
    assert scores.grad.abs().sum() > 0
