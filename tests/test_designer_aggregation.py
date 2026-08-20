"""Tests for request-driven aggregation selection (the avg->SUM fix).

The designer used to infer aggregation from the column name only, so "average order value"
became SUM(amount). These pin the two new entry points: the explicit mapper field
(normalize_requested_agg) and the deterministic NL-verb fallback (agg_from_request).
"""
from __future__ import annotations

from semanticflow.agents.designer import agg_from_request, normalize_requested_agg


class TestNormalizeRequestedAgg:
    def test_synonyms_map_to_canonical(self):
        assert normalize_requested_agg("average") == "average"
        assert normalize_requested_agg("AVG") == "average"
        assert normalize_requested_agg("mean") == "average"
        assert normalize_requested_agg("distinct") == "count_distinct"
        assert normalize_requested_agg("total") == "sum"

    def test_ratio_is_not_a_measure_agg(self):
        assert normalize_requested_agg("ratio") is None
        assert normalize_requested_agg("percentage") is None

    def test_missing_or_nonstring_is_none(self):
        assert normalize_requested_agg(None) is None
        assert normalize_requested_agg(123) is None
        assert normalize_requested_agg("") is None


class TestAggFromRequest:
    def test_average_request(self):
        assert agg_from_request("Show average order value per month.") == "average"

    def test_count_request(self):
        assert agg_from_request("How many orders has each customer placed?") == "count"

    def test_distinct_beats_count(self):
        assert agg_from_request("How many distinct customers ordered?") == "count_distinct"

    def test_max_request(self):
        assert agg_from_request("What is the highest order amount?") == "max"

    def test_percentage_returns_none(self):
        # ratio metric — must not pick a wrong measure agg
        assert agg_from_request("What percentage of orders use coupons?") is None

    def test_no_verb_returns_none(self):
        # "Show me the numbers" implies nothing — keep column-name inference
        assert agg_from_request("Show me the numbers.") is None

    def test_empty_input(self):
        assert agg_from_request("") is None
        assert agg_from_request(None) is None
