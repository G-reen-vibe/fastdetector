"""Tests for metric direction and degenerate-input conventions."""

import numpy as np
import pytest

from fastdetector.visualization.metrics import compute_classifier_metrics


def _separable():
    """Human scores low, AI scores high; arrays are (values, is_positive)."""
    human = np.array([0.1, 0.15, 0.2, 0.05])
    ai = np.array([0.8, 0.85, 0.9, 0.95])
    return [human, ai], [False, True]


def test_auroc_is_high_for_higher_is_ai():
    arrays, labels = _separable()
    m = compute_classifier_metrics(arrays, labels, threshold=0.5, flip_inequality=False)
    assert m["auroc"] == pytest.approx(1.0)


def test_auroc_respects_lower_is_ai_direction():
    """A lower_is_ai classifier that separates perfectly must score ~1.0.

    Previously the raw scores went to roc_auc_score regardless of direction,
    so this returned ~0.0 and the 'best classifier' highlighting in the
    comparison table selected the worst one.
    """
    human = np.array([0.8, 0.85, 0.9, 0.95])
    ai = np.array([0.1, 0.15, 0.2, 0.05])
    m = compute_classifier_metrics([human, ai], [False, True],
                                   threshold=0.5, flip_inequality=True)
    assert m["auroc"] == pytest.approx(1.0)


def test_auroc_and_accuracy_agree_on_direction():
    """Both should indicate a good classifier under either convention."""
    for flip in (False, True):
        human = np.array([0.8, 0.9]) if flip else np.array([0.1, 0.2])
        ai = np.array([0.1, 0.2]) if flip else np.array([0.8, 0.9])
        m = compute_classifier_metrics([human, ai], [False, True],
                                       threshold=0.5, flip_inequality=flip)
        assert m["acc"] == pytest.approx(1.0)
        assert m["auroc"] > 0.5


def test_bertscore_empty_text_is_maximally_distant():
    """Empty/failed generations must not score as identical to the source."""
    torch = pytest.importorskip("torch")
    from fastdetector.statistics.statistics_embedding import bertscore

    src = torch.nn.functional.normalize(torch.rand(5, 8), p=2, dim=1)
    empty = torch.empty((0, 8))

    prec, rec, f1 = bertscore([src], [empty])
    assert prec == [1.0] and rec == [1.0] and f1 == [1.0]


def test_bertscore_identical_text_is_near_zero_distance():
    torch = pytest.importorskip("torch")
    from fastdetector.statistics.statistics_embedding import bertscore

    src = torch.nn.functional.normalize(torch.rand(6, 8), p=2, dim=1)
    prec, rec, f1 = bertscore([src], [src])
    assert f1[0] == pytest.approx(0.0, abs=1e-5)
    assert f1[0] < 1.0, "identical text must be closer than empty text"
