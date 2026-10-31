import numpy as np
import pytest

from newkcat.metrics import regression_metrics


def test_metrics_detect_float_overflow():
    with pytest.raises(ValueError, match="overflowed"):
        regression_metrics([1e308], [-1e308])


def test_exact_prediction_and_tied_spearman():
    result = regression_metrics([1, 2, 2, 5], [1, 2, 2, 5])
    assert result["rmse"] == 0
    assert result["r2"] == 1
    assert result["pearson"] == pytest.approx(1)
    assert result["spearman"] == pytest.approx(1)
    assert regression_metrics([1, 2, 2, 5], [5, 2, 2, 1])["spearman"] == pytest.approx(-1)


def test_undefined_values_are_none():
    result = regression_metrics([1, 1], [3, 4])
    assert result["r2"] is None
    assert result["pearson"] is None
    assert result["spearman"] is None
    assert regression_metrics([1], [3])["rmse"] == 2


@pytest.mark.parametrize("target,prediction", [
    ([[1], [2]], [1, 2]), ([1, 2], [1]), ([], []), ([np.nan], [0]), ([0], [np.inf]),
])
def test_metrics_reject_broadcast_or_nonfinite(target, prediction):
    with pytest.raises(ValueError):
        regression_metrics(target, prediction)
