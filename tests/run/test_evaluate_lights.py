import pytest

from evaluate_lights import auroc, light_of, summarize


def test_light_of():
    assert light_of("L3_piece-07_shot-01.png") == "L3"
    assert light_of("piece-07_shot-01.png") == "-"


def test_summarize_counts_each_light_apart():
    rows = [("L1", "good", 0.1), ("L1", "good", 0.3), ("L1", "marker", 0.5),
            ("L2", "good", 0.1), ("L2", "lovitura", 0.1)]
    s = summarize(rows, threshold=0.2)
    assert s["L1"]["false_alarms"] == 1 and s["L1"]["false_alarm_rate"] == 0.5
    assert s["L1"]["visible_rate"] == 1.0
    assert s["L2"]["detected"]["lovitura"] == [0, 1] and s["L2"]["visible_rate"] is None
    assert s["toate"]["good"] == 3 and s["toate"]["detected"]["marker"] == [1, 1]


def test_a_score_equal_to_the_threshold_is_ok():
    s = summarize([("L1", "good", 0.2), ("L1", "marker", 0.2)], threshold=0.2)
    assert s["L1"]["false_alarms"] == 0 and s["L1"]["detected"]["marker"] == [0, 1]


def test_auroc_counts_ties_as_half():
    assert auroc([0.1, 0.2], [0.3, 0.4]) == 1.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([0.1], []) is None
