"""Tests for three execution-grounded designer/mapper fixes.

1. Day-grain declaration (real_017, synth_009): the declared `time_granularity` of a
   time dimension is the DATA grain ('day'), never the QUERY granularity. Declaring
   at e.g. 'month' makes MetricFlow offer only `metric_time__month`, so a
   `{{ TimeDimension('metric_time', 'day') }}` filter cannot resolve. The query-time
   grain is passed separately by the executor to `mf query`.
2. Consensus filter pile-up (real_017, synth_009): `consensus_vote` must take the
   FILTERS from the top-confidence provider's spec (like order_by/limit), not union
   every >=0.7*max variant — that ANDed 6-8 semantically-equivalent filters together.
3a. Synthetic `ds` collision on joined models (real_015): no synthetic `ds` time
    dimension, defaults, or primary_entity on measure-less secondary models, or dbt
    parse fails with "Duplicate dimension + primary entity pairing".
3b. Measure/dimension name collision (real_011): a measure whose name collides with
    a dimension in the same model is renamed by suffixing its aggregation
    (`customer_id` -> `customer_id_count`); the metric references the renamed measure.
"""
from __future__ import annotations

import asyncio

from semanticflow.agents.designer import (
    DesignerAgent,
    _dimension_for,
    _resolve_measure_dimension_collisions,
)
from semanticflow.agents.semantic_mapper import consensus_vote
from semanticflow.agents.types import AgentMessage
from semanticflow.dbt_integration.semantic_yaml import (
    DimensionSpec,
    MeasureSpec,
    semantic_model_to_dict,
)
from semanticflow.tasks.loader import Task


def _run_designer(spec: dict, schema_context: dict, project_dir: str,
                  nl_request: str = "build the metric") -> dict:
    """Run the designer offline (no LLM client) and return its design proposals."""
    task = Task(
        task_id="fix_test",
        nl_request=nl_request,
        schema_context=schema_context,
        dbt_project=project_dir,
    )
    out = asyncio.run(DesignerAgent().run(AgentMessage(task=task, state={"semantic_spec": spec})))
    return out.state["design_proposals"]


class TestDayGrainDeclaration:
    """Fix 1: declared time-dimension grain is always 'day' (the data grain)."""

    def test_dimension_for_declares_day_grain(self):
        dim = _dimension_for("order_date", {"order_date"}, {"order_date": "date"})
        assert dim is not None and dim.type == "time"
        assert dim.type_params == {"time_granularity": "day"}

    def test_month_query_granularity_not_copied_into_declaration(self, tmp_path):
        # real_017-style: monthly query over daily order_date. The old code declared
        # the dim at 'month', so a day-grain TimeDimension filter could not resolve.
        schema = {
            "models": ["orders"],
            "columns": {"orders": ["order_id", "customer_id", "order_date", "amount"]},
            "data_types": {"orders": {"order_date": "date", "amount": "integer"}},
        }
        spec = {"source_model": "orders", "metric_name": "revenue",
                "base_measures": ["amount"], "group_by": ["order_date"],
                "time_granularity": "month", "filters": []}
        design = _run_designer(spec, schema, str(tmp_path))
        dims = design["semantic_models"][0]["dimensions"]
        time_dims = [d for d in dims if d["type"] == "time"]
        assert time_dims, "expected a time dimension on the primary model"
        for d in time_dims:
            assert d["type_params"]["time_granularity"] == "day"
        # The spine generated alongside is daily as well.
        assert design["time_spine"]["time_granularity"] == "day"

    def test_metric_time_binding_declares_day_grain(self, tmp_path):
        # When group_by uses the virtual 'metric_time', the bound real column is
        # likewise declared at day grain regardless of the query granularity.
        schema = {
            "models": ["orders"],
            "columns": {"orders": ["order_id", "order_date", "amount"]},
            "data_types": {"orders": {"order_date": "date", "amount": "integer"}},
        }
        spec = {"source_model": "orders", "metric_name": "revenue",
                "base_measures": ["amount"], "group_by": ["metric_time"],
                "time_granularity": "quarter", "filters": []}
        design = _run_designer(spec, schema, str(tmp_path))
        time_dims = [d for d in design["semantic_models"][0]["dimensions"] if d["type"] == "time"]
        assert time_dims and all(
            d["type_params"]["time_granularity"] == "day" for d in time_dims
        )


class TestConsensusFilterSelection:
    """Fix 2: filters come from the top-confidence spec, not a cross-provider union."""

    _VARIANTS = [
        ["{{ TimeDimension('metric_time', 'day') }} < '2018-04-01'"],
        ["order_date < date '2018-04-01'"],
        ["{{ TimeDimension('metric_time', 'month') }} = '2018-02'"],
    ]

    def _specs(self):
        return [
            {"metric_name": "revenue", "base_measures": ["amount"],
             "group_by": ["order_date"], "filters": list(f), "time_granularity": "month"}
            for f in self._VARIANTS
        ]

    def test_filters_taken_from_top_confidence_spec(self):
        out = consensus_vote(self._specs(), [0.9, 0.8, 0.7])
        assert out["filters"] == self._VARIANTS[0]

    def test_no_pileup_of_equivalent_variants(self):
        # Old behaviour kept every variant scoring >= 0.7*max, ANDing 3+ equivalent
        # filters together. Exactly one variant must survive here.
        out = consensus_vote(self._specs(), [0.9, 0.8, 0.7])
        assert len(out["filters"]) == 1

    def test_top_confidence_is_selected_not_first(self):
        out = consensus_vote(self._specs(), [0.7, 0.9, 0.6])
        assert out["filters"] == self._VARIANTS[1]

    def test_measures_and_group_by_still_unioned(self):
        # Only filters change selection strategy; list fields where union is correct
        # keep the vote behaviour.
        specs = self._specs()
        specs[1]["base_measures"] = ["amount", "order_total"]
        specs[2]["base_measures"] = ["amount", "order_total"]
        # order_total scores 1.8 >= 0.7 * max(2.3), so the vote keeps the union.
        out = consensus_vote(specs, [0.5, 0.9, 0.9])
        assert out["base_measures"] == ["amount", "order_total"]
        assert out["group_by"] == ["order_date"]

    def test_empty_filters_on_top_spec(self):
        specs = self._specs()
        specs[0]["filters"] = []
        out = consensus_vote(specs, [0.9, 0.5, 0.5])
        assert out["filters"] == []

    def test_missing_filters_key_tolerated(self):
        out = consensus_vote([{"metric_name": "m", "base_measures": ["amount"]}], [0.8])
        assert out["filters"] == []


class TestSecondaryModelNoSyntheticDs:
    """Fix 3a: measure-less secondary models get no synthetic 'ds'/defaults/primary_entity."""

    _SCHEMA = {
        "models": ["stg_payments", "orders"],
        "columns": {
            "stg_payments": ["payment_id", "order_id", "payment_method", "amount"],
            "orders": ["order_id", "customer_id", "status"],
        },
        "data_types": {
            "stg_payments": {"amount": "integer"},
            "orders": {"status": "varchar"},
        },
    }
    _SPEC = {"source_model": "stg_payments", "metric_name": "revenue",
             "base_measures": ["amount"], "group_by": ["status"], "filters": []}

    def _design(self, tmp_path):
        return _run_designer(self._SPEC, self._SCHEMA, str(tmp_path))

    def test_secondary_model_has_no_ds_dimension_or_defaults(self, tmp_path):
        design = self._design(tmp_path)
        models = {m["name"]: m for m in design["semantic_models"]}
        secondary = models["orders"]
        assert secondary["measures"] == []
        assert [d["name"] for d in secondary["dimensions"]] == ["status"]
        assert secondary["defaults"] is None
        assert secondary["primary_entity"] is None

    def test_primary_model_keeps_synthetic_ds(self, tmp_path):
        # stg_payments has measures but no real time column, so it still needs the
        # synthetic agg_time_dimension; only the secondary side is suppressed.
        design = self._design(tmp_path)
        primary = design["semantic_models"][0]
        assert primary["name"] == "stg_payments"
        assert any(d["name"] == "ds" and d["type"] == "time" for d in primary["dimensions"])
        assert primary["defaults"] == {"agg_time_dimension": "ds"}

    def test_no_duplicate_entity_time_dimension_pairing(self, tmp_path):
        # The dbt parse error was: (`order`, `ds`) declared on BOTH models. Only one
        # model may carry the synthetic time dimension now.
        design = self._design(tmp_path)
        models_with_ds = [
            m["name"] for m in design["semantic_models"]
            if any(d["name"] == "ds" for d in m["dimensions"])
        ]
        assert models_with_ds == ["stg_payments"]

    def test_join_keys_still_carried_by_entities(self, tmp_path):
        # Unit-level join check: MetricFlow resolves the join through the entities
        # list — the primary model exposes `order` as a foreign entity and the
        # secondary declares it as its `type: primary` entity. The rendered secondary
        # YAML needs no `primary_entity` key (codegen strips it whenever a
        # type-primary entity exists, so the YAML is unchanged by passing None).
        design = self._design(tmp_path)
        models = {m["name"]: m for m in design["semantic_models"]}
        primary_foreign = {e["name"] for e in models["stg_payments"]["entities"]
                           if e["type"] == "foreign"}
        assert "order" in primary_foreign
        secondary_primary = [e for e in models["orders"]["entities"] if e["type"] == "primary"]
        assert len(secondary_primary) == 1
        assert secondary_primary[0]["name"] == "order"
        assert secondary_primary[0]["expr"] == "order_id"

        from semanticflow.dbt_integration.semantic_yaml import SemanticModelSpec
        rendered = semantic_model_to_dict(SemanticModelSpec(**models["orders"]))
        assert "primary_entity" not in rendered
        assert any(e.get("type") == "primary" for e in rendered["entities"])


class TestMeasureDimensionCollision:
    """Fix 3b: colliding measure names are renamed with their aggregation suffix."""

    def test_colliding_measures_renamed_with_agg_suffix(self):
        measures = [
            MeasureSpec(name="customer_id", agg="count", expr="customer_id"),
            MeasureSpec(name="customer_lifetime_value", agg="sum", expr="customer_lifetime_value"),
            MeasureSpec(name="amount", agg="sum", expr="amount"),
        ]
        dimensions = [
            DimensionSpec(name="customer_id"),
            DimensionSpec(name="customer_lifetime_value"),
            DimensionSpec(name="status"),
        ]
        out = _resolve_measure_dimension_collisions(measures, dimensions)
        assert [m.name for m in out] == [
            "customer_id_count", "customer_lifetime_value_sum", "amount",
        ]
        # expr keeps pointing at the original columns
        assert out[0].expr == "customer_id"
        assert out[1].expr == "customer_lifetime_value"

    def test_renamed_measure_without_expr_gains_original_column(self):
        out = _resolve_measure_dimension_collisions(
            [MeasureSpec(name="status", agg="count", expr=None)],
            [DimensionSpec(name="status")],
        )
        assert out[0].name == "status_count"
        assert out[0].expr == "status"

    def test_suffixed_name_also_colliding_gets_disambiguated(self):
        out = _resolve_measure_dimension_collisions(
            [MeasureSpec(name="customer_id", agg="count", expr="customer_id")],
            [DimensionSpec(name="customer_id"), DimensionSpec(name="customer_id_count")],
        )
        assert out[0].name == "customer_id_count_measure"

    def test_no_collision_is_a_noop(self):
        measures = [MeasureSpec(name="amount", agg="sum", expr="amount")]
        out = _resolve_measure_dimension_collisions(measures, [DimensionSpec(name="status")])
        assert out == measures

    def test_designer_emits_renamed_measure_and_updates_metric(self, tmp_path):
        # real_011-style: consensus merged one provider's measure (customer_id) with
        # another's group-by (customer_id) — dbt rejected the same name as both
        # MEASURE and DIMENSION. The measure is renamed; the metric follows it.
        schema = {
            "models": ["customers"],
            "columns": {"customers": ["customer_id", "customer_lifetime_value",
                                      "number_of_orders"]},
            "data_types": {},
        }
        spec = {"source_model": "customers", "metric_name": "customer_count",
                "base_measures": ["customer_id"], "group_by": ["customer_id"],
                "filters": []}
        design = _run_designer(spec, schema, str(tmp_path))
        model = design["semantic_models"][0]
        measure_names = [m["name"] for m in model["measures"]]
        dimension_names = [d["name"] for d in model["dimensions"]]
        assert "customer_id" in dimension_names
        assert "customer_id" not in measure_names
        assert "customer_id_count" in measure_names
        renamed = next(m for m in model["measures"] if m["name"] == "customer_id_count")
        assert renamed["expr"] == "customer_id"
        assert design["metrics"][0]["type_params"]["measure"] == "customer_id_count"
