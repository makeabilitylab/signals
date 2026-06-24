"""Unit tests for gesturerec.experiments.TrialClassificationResult.

Exercises the n-best-list sorting and is_correct contract with lightweight stub
trials, so no real classifier run or data loading is needed.
"""
from gesturerec.experiments import TrialClassificationResult


class _StubTrial:
    """Minimal stand-in for a Trial: only the attributes the result class touches."""

    def __init__(self, gesture_name, trial_num=0):
        self.gesture_name = gesture_name
        self.trial_num = trial_num

    def get_ground_truth_gesture_name(self):
        return self.gesture_name


def test_nbest_list_sorted_ascending_by_score_and_closest_is_lowest():
    test_trial = _StubTrial("Shake")
    good = _StubTrial("Shake")
    bad = _StubTrial("Wave")
    # Lower score == closer match (the algorithms return distances).
    result = TrialClassificationResult(test_trial, [(bad, 9.0), (good, 1.0)])

    assert result.n_best_list_sorted[0][1] == 1.0
    assert result.closest_trial is good
    assert result.score == 1.0


def test_is_correct_true_when_closest_matches_ground_truth():
    test_trial = _StubTrial("Shake")
    result = TrialClassificationResult(
        test_trial, [(_StubTrial("Shake"), 2.0), (_StubTrial("Wave"), 5.0)])
    assert result.is_correct is True


def test_is_correct_false_when_closest_is_wrong_gesture():
    test_trial = _StubTrial("Shake")
    result = TrialClassificationResult(
        test_trial, [(_StubTrial("Wave"), 0.5), (_StubTrial("Shake"), 5.0)])
    assert result.is_correct is False


def test_correct_match_index_in_nbest_list():
    test_trial = _StubTrial("Shake")
    # Closest is Wave (0.5); the correct Shake template is next (index 1).
    result = TrialClassificationResult(
        test_trial, [(_StubTrial("Wave"), 0.5), (_StubTrial("Shake"), 5.0)])
    assert result.get_correct_match_index_nbestlist() == 1
