"""Tests of the original fixed rule baseline, not benchmark accuracy claims."""

import pytest

from src.evaluation_baseline import predict_baseline


@pytest.mark.parametrize("text,label", [
    ("Revenue grew and profit improved.", "positive"), ("Losses and lower sales.", "negative"),
    ("The board met on Tuesday.", "neutral"), ("Profit and loss.", "neutral"),
    ("Profit did not increase.", "neutral"),  # One positive match and its negated counterpart.
    ("Revenue did not grow.", "negative"), ("There were no losses.", "positive"),
    ("We never saw growth.", "negative"), ("Results didn't improve.", "negative"),
    ("Not expected; growth returned.", "positive"), ("REVENUE GREW.", "positive"),
])
def test_fixed_baseline_rules(text, label):
    assert predict_baseline(text) == label
    assert predict_baseline(text) == label


@pytest.mark.parametrize("text", [None, "", " \n"])
def test_invalid_baseline_input(text):
    with pytest.raises(ValueError):
        predict_baseline(text)
