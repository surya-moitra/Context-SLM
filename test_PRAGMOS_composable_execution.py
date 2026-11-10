import copy
from dataclasses import replace
from decimal import Decimal
import itertools
import unittest

from PRAGMOS_composable_execution import (
    AliasBinding, ContractError, Quantity, Source, execute_plan, ground_fact,
    parse_measurement_span, source_measurements, validate_plan,
)


def operand(key="O1", *, actor="user", cardinality="collection", statuses=None):
    return {"id": key, "description": "supported source facts", "actor": actor,
            "statuses": statuses or ["completed"], "cardinality": cardinality, "kinds": ["event"]}


def step(key, op, inputs=(), **params):
    return {"id": key, "op": op, "inputs": list(inputs), "params": params}


def plan(steps, *, operands=None, column="value", columns=None, shape="scalar", unit=None, digits=None):
    return {"version": 1, "applicable": True, "operands": operands or [operand()], "steps": steps,
            "output": {"step": steps[-1]["id"], "columns": columns or [column], "shape": shape,
                       "unit": unit, "decimals": digits}}


def data(values, *, measurement_unit="dollars", roles=None, operands=None, speaker="user"):
    sources, facts = {}, []
    for index, value in enumerate(values, 1):
        suffix = ""
        fields = roles[index - 1] if roles else {}
        if fields:
            suffix = " at " + " with ".join(fields.values())
        quote = f"I paid {value} {measurement_unit}{suffix}."
        source = Source(f"S{index}", f"s{index}", f"t{index}", "user", speaker, "2027-01-02", quote)
        sources[source.source_id] = source
        facts.append(ground_fact({
            "source_id": source.source_id, "quote": quote, "actor_span": "I", "predicate": "paid",
            "status": "completed", "kind": "event", "operands": operands or ["O1"],
            "roles": fields,
            "measurement": {"text": f"{value} {measurement_unit}", "value": str(value), "unit": measurement_unit},
        }, sources, set(operands or ["O1"])))
    return sources, facts


def scan(key="a", source="O1"):
    return step(key, "source", operand=source)


def total(source="a", key="b", keys=None, aggregate="sum"):
    return step(key, "reduce", [source], keys=keys or [], aggregate=aggregate, column="value")


class ComposableExecutionTests(unittest.TestCase):
    def test_group_sum_before_rank_not_largest_transaction(self):
        sources, facts = data([17, 38, 29], roles=[{"location": "Cedar"}, {"location": "Birch"}, {"location": "Cedar"}])
        request = plan([scan(), total(keys=["location"]), step("c", "rank", ["b"], column="value", direction="max")], column="location")
        for ordering in itertools.permutations(facts):
            result = execute_plan(request, ordering, sources)
            self.assertEqual(result["answer"], "Cedar")
            self.assertEqual(len(result["provenance"]), 2)
            self.assertEqual(result["steps"][1]["row_count"], 2)

    def test_individual_extremum_keeps_different_meaning(self):
        sources, facts = data([17, 38, 29], roles=[{"location": "Cedar"}, {"location": "Birch"}, {"location": "Cedar"}])
        request = plan([scan(), step("b", "rank", ["a"], column="value", direction="max")], column="location")
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "Birch")

    def test_numeric_distinct_uses_canonical_value_not_decimal_spelling(self):
        sources, facts = data(["5", "5.0", "5.00", "6"])
        request = plan([scan(), step("b", "distinct", ["a"], keys=["value"]),
                        total("b", "c", aggregate="count")])
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "2")
        self.assertEqual(execute_plan(plan([scan(), total(aggregate="count")]), facts, sources)["answer"], "4")

    def test_numeric_distinct_normalizes_convertible_units_not_dimensions(self):
        sources, facts = data(["1.0"], measurement_unit="hours")
        other_sources, other_facts = data(["60"], measurement_unit="minutes")
        source = replace(other_sources["S1"], source_id="S2", session_id="s2", turn_id="t2")
        sources["S2"] = source
        fact = other_facts[0]
        facts.append(ground_fact({"source_id": "S2", "quote": source.text, "actor_span": "I",
                                 "predicate": "paid", "status": "completed", "kind": "event", "operands": ["O1"],
                                 "roles": {}, "measurement": {"text": fact.raw_measurement,
                                                               "value": fact.raw_value, "unit": fact.raw_unit}}, sources, {"O1"}))
        request = plan([scan(), step("b", "distinct", ["a"], keys=["value"]),
                        total("b", "c", aggregate="count")])
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "1")
        money_sources, money_facts = data(["3600"])
        source = replace(money_sources["S1"], source_id="S3", session_id="s3", turn_id="t3")
        sources["S3"] = source
        fact = money_facts[0]
        facts.append(ground_fact({"source_id": "S3", "quote": source.text, "actor_span": "I",
                                 "predicate": "paid", "status": "completed", "kind": "event", "operands": ["O1"],
                                 "roles": {}, "measurement": {"text": fact.raw_measurement,
                                                               "value": fact.raw_value, "unit": fact.raw_unit}}, sources, {"O1"}))
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "2")

    def test_grouping_is_independent_of_names_and_amount_scale(self):
        for names in (("North", "South"), ("Alpha Labs", "Beta Works")):
            for factor in (1, 10, 100):
                sources, facts = data([2 * factor, 3 * factor, 2 * factor],
                                      roles=[{"location": names[0]}, {"location": names[1]}, {"location": names[0]}])
                request = plan([scan(), total(keys=["location"]), step("c", "rank", ["b"], column="value", direction="max")], column="location")
                self.assertEqual(execute_plan(request, facts, sources)["answer"], names[0])

    def test_equal_group_totals_do_not_choose_arbitrary_winner(self):
        sources, facts = data([2, 4, 2], roles=[{"location": "North"}, {"location": "South"}, {"location": "North"}])
        request = plan([scan(), total(keys=["location"]), step("c", "rank", ["b"], column="value", direction="max")], column="location")
        result = execute_plan(request, facts, sources)
        self.assertFalse(result["executable"])
        self.assertEqual(result["reason"], "output_not_unique")

    def test_converted_value_does_not_need_to_appear_in_source(self):
        sources, facts = data([3], measurement_unit="hours")
        result = execute_plan(plan([scan(), total()], unit="minutes"), facts, sources)
        self.assertEqual(result["answer"], "180 minutes")
        self.assertEqual(result["provenance"][0]["raw_value"], "3")
        self.assertEqual(result["provenance"][0]["quantity"]["canonical_value"], "10800")

    def test_mixed_convertible_units_preserve_requested_unit(self):
        sources, facts = data([3], measurement_unit="hours")
        source = Source("S2", "s2", "t2", "user", "user", None, "I worked for 300 minutes.")
        sources["S2"] = source
        facts.append(ground_fact({"source_id": "S2", "quote": source.text, "actor_span": "I", "predicate": "worked",
                                 "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                                 "measurement": {"text": "300 minutes", "value": "300", "unit": "minutes"}}, sources, {"O1"}))
        self.assertEqual(execute_plan(plan([scan(), total()], unit="hours"), facts, sources)["answer"], "8 hours")

    def test_mass_and_distance_abbreviations_are_not_dropped(self):
        for first, second, target, expected in (("kg", "g", "kg", "8 kg"), ("km", "m", "km", "8 km")):
            sources, facts = data([3], measurement_unit=first)
            source = Source("S2", "s2", "t2", "user", "user", None, f"I measured 5000{second}.")
            sources["S2"] = source
            facts.append(ground_fact({"source_id": "S2", "quote": source.text, "actor_span": "I", "predicate": "measured",
                                     "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                                     "measurement": {"text": f"5000{second}", "value": "5000", "unit": second}}, sources, {"O1"}))
            self.assertEqual(execute_plan(plan([scan(), total()], unit=target), facts, sources)["answer"], expected)

    def test_incompatible_output_dimension_is_not_erased(self):
        sources, facts = data([3], measurement_unit="kg")
        result = execute_plan(plan([scan(), total()], unit="minutes"), facts, sources)
        self.assertEqual(result["reason"], "incompatible_units")

    def test_calendar_months_are_not_fixed_elapsed_seconds(self):
        sources, facts = data([2], measurement_unit="months")
        result = execute_plan(plan([scan(), total()], unit="days"), facts, sources)
        self.assertEqual(result["reason"], "incompatible_units")

    def test_collection_allows_many_facts_scalar_does_not(self):
        sources, facts = data([17, 29])
        self.assertEqual(execute_plan(plan([scan(), total()]), facts, sources)["answer"], "$46")
        request = plan([scan(), total()], operands=[operand(cardinality="scalar")])
        self.assertEqual(execute_plan(request, facts, sources)["reason"], "ambiguous_scalar_operand:O1")

    def test_missing_scalar_is_not_zero(self):
        sources, facts = data([17])
        request = plan([scan(), scan("b", "O2"), step("c", "arithmetic", ["a", "b"], operator="subtract", left="value", right="value")],
                       operands=[operand(cardinality="scalar"), operand("O2", cardinality="scalar")])
        result = execute_plan(request, facts, sources)
        self.assertEqual(result["reason"], "missing_operand:O2")
        self.assertIsNone(result["answer"])

    def test_duplicate_assertion_is_not_double_counted(self):
        sources, facts = data([12, 12])
        result = execute_plan(plan([scan(), total()]), facts + facts, sources)
        self.assertEqual(result["answer"], "$24")
        self.assertEqual(len(result["provenance"]), 2)

    def test_exact_decimal_arithmetic_and_zero(self):
        sources, facts = data(["0.1", "0.2", "0"])
        self.assertEqual(execute_plan(plan([scan(), total()]), facts, sources)["answer"], "$0.3")

    def test_actor_swap_cannot_supply_missing_operand(self):
        sources, facts = data([17], speaker="Asha")
        request = plan([scan(), total()], operands=[operand(actor="Mira")])
        self.assertEqual(execute_plan(request, facts, sources)["reason"], "missing_operand:O1")

    def test_non_event_state_cannot_be_added_as_transaction(self):
        sources, facts = data([17])
        facts = [replace(facts[0], kind="state")]
        # A forged kind also changes the source-backed fact identity.
        self.assertFalse(execute_plan(plan([scan(), total()]), facts, sources)["executable"])

    def test_source_qualification_survives_compacted_quote(self):
        compact = "I paid 17 dollars for supplies"
        source = Source("S1", "s1", "t1", "user", "user", None,
                        compact + ", but that amount was only planned and was never paid.")
        sources = {"S1": source}
        fact = ground_fact({"source_id": "S1", "quote": compact, "actor_span": "I", "predicate": "paid",
                            "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                            "measurement": {"text": "17 dollars", "value": "17", "unit": "dollars"}}, sources, {"O1"})
        self.assertNotEqual(fact.status, "completed")
        self.assertFalse(execute_plan(plan([scan(), total()]), [fact], sources)["executable"])

    def test_distinct_projects_requested_identity_not_claim_id(self):
        sources, facts = data([1, 1, 1], roles=[{"weekday": "Monday"}, {"weekday": "Monday"}, {"weekday": "Friday"}])
        request = plan([scan(), step("b", "project", ["a"], columns={"day": "weekday"}),
                        step("c", "distinct", ["b"], keys=["day"]), total("c", "d", aggregate="count")])
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "2")

    def test_explicit_role_scoped_aliases_do_not_mutate_inputs(self):
        sources, facts = data([17, 38, 29], roles=[{"location": "Cedar Shop"}, {"location": "Birch"}, {"location": "Cedar"}])
        sources["S4"] = Source("S4", "s4", "t4", "user", "user", None, "I call Cedar Shop Cedar.")
        aliases = [AliasBinding("location", "Cedar Shop", "Cedar", "S4", sources["S4"].text, "call")]
        before = copy.deepcopy((sources, facts, aliases))
        request = plan([scan(), total(keys=["location"]), step("c", "rank", ["b"], column="value", direction="max")], column="location")
        self.assertEqual(execute_plan(request, facts, sources, aliases=aliases)["answer"], "Cedar")
        self.assertEqual((sources, facts, aliases), before)

    def test_derived_unit_price_has_dimensional_provenance(self):
        sources, facts = data([12], operands=["PAYMENT"])
        source = Source("S2", "s2", "t2", "user", "user", None, "I received 4 items.")
        sources["S2"] = source
        facts.append(ground_fact({"source_id": "S2", "quote": source.text, "actor_span": "I", "predicate": "received",
                                 "status": "completed", "kind": "event", "operands": ["ITEMS"], "roles": {},
                                 "measurement": {"text": "4 items", "value": "4", "unit": "items"}}, sources, {"PAYMENT", "ITEMS"}))
        request = plan([scan("a", "PAYMENT"), scan("b", "ITEMS"),
                        step("c", "arithmetic", ["a", "b"], operator="divide", left="value", right="value")],
                       operands=[operand("PAYMENT", cardinality="scalar"), operand("ITEMS", cardinality="scalar")], unit="usd/item")
        result = execute_plan(request, facts, sources)
        self.assertEqual(result["answer"], "3 usd/item")
        self.assertEqual(len(result["provenance"]), 2)

    def test_zero_denominator_is_not_calculated(self):
        sources, facts = data([12], operands=["A", "B"])
        zero_sources, zero_facts = data([0], operands=["B"])
        # Place the second assertion in a distinct real source.
        other = replace(zero_sources["S1"], source_id="S2", session_id="s2", turn_id="t2")
        sources["S2"] = other
        raw = zero_facts[0]
        facts[0] = ground_fact({"source_id": "S1", "quote": facts[0].quote, "actor_span": "I", "predicate": "paid",
                               "status": "completed", "kind": "event", "operands": ["A"], "roles": {},
                               "measurement": {"text": facts[0].raw_measurement, "value": "12", "unit": "dollars"}}, sources, {"A", "B"})
        facts.append(ground_fact({"source_id": "S2", "quote": raw.quote, "actor_span": "I", "predicate": "paid",
                                 "status": "completed", "kind": "event", "operands": ["B"], "roles": {},
                                 "measurement": {"text": raw.raw_measurement, "value": "0", "unit": "dollars"}}, sources, {"A", "B"}))
        request = plan([scan("a", "A"), scan("b", "B"), step("c", "arithmetic", ["a", "b"], operator="divide", left="value", right="value")],
                       operands=[operand("A", cardinality="scalar"), operand("B", cardinality="scalar")])
        self.assertEqual(execute_plan(request, facts, sources)["reason"], "zero_denominator")

    def test_forged_canonical_value_cannot_be_certified(self):
        sources, facts = data([3])
        altered = replace(facts[0], quantity=Quantity(Decimal(999), facts[0].quantity.dimensions, "usd"))
        result = execute_plan(plan([scan(), total()]), [altered], sources)
        self.assertEqual(result["reason"], "fact_not_equal_to_grounded_source")

    def test_plan_cannot_hide_mandatory_operand_in_unused_step(self):
        request = plan([scan(), scan("b", "O2"), total("a", "c")], operands=[operand(), operand("O2")])
        with self.assertRaisesRegex(ContractError, "unused_plan_step"):
            validate_plan(request)

    def test_cycles_and_arbitrary_code_are_rejected(self):
        for request in (plan([step("a", "source", ["a"], operand="O1")]),
                        plan([step("a", "eval", expression="__import__('os')")])):
            with self.assertRaises(ContractError):
                validate_plan(request)

    def test_invalid_raw_units_and_nonfinite_values_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity", True):
            with self.assertRaises(ContractError):
                Quantity.from_raw(value, "kg")
        with self.assertRaises(ContractError):
            Quantity.from_raw(1, "invented-units")

    def test_schema_validity_is_not_claimed_as_semantic_equivalence(self):
        sources, facts = data([3])
        result = execute_plan(plan([scan(), total()]), facts, sources)
        self.assertTrue(result["executable"])
        self.assertFalse(result["semantic_equivalence_verified"])

    def test_percentage_change_is_composed_from_supported_inputs(self):
        sources, facts = data([100, 80], operands=["BEFORE", "AFTER"])
        facts = [replace(facts[0], operands=("BEFORE",)), replace(facts[1], operands=("AFTER",))]
        request = plan([scan("a", "BEFORE"), scan("b", "AFTER"),
                        step("c", "arithmetic", ["a", "b"], operator="subtract", left="value", right="value"),
                        step("d", "arithmetic", ["c", "a"], operator="divide", left="value", right="value")],
                       operands=[operand("BEFORE", cardinality="scalar"), operand("AFTER", cardinality="scalar")], unit="percent")
        result = execute_plan(request, facts, sources)
        self.assertEqual(result["answer"], "20%")
        self.assertEqual(len(result["provenance"]), 2)

    def test_per_entity_derived_values_can_be_compared_without_losing_identity(self):
        sources, facts = data([100, 80, 60, 42], operands=["QUOTED", "PAID"],
                              roles=[{"location": "North"}, {"location": "North"},
                                     {"location": "South"}, {"location": "South"}])
        facts = [replace(fact, operands=("QUOTED" if i % 2 == 0 else "PAID",)) for i, fact in enumerate(facts)]
        request = plan([scan("a", "QUOTED"), scan("b", "PAID"),
                        step("c", "join", ["a", "b"], on=[["location", "location"]], cardinality="one_to_one"),
                        step("d", "calculate", ["c"], operator="subtract", left="left.value", right="right.value", column="saving"),
                        step("e", "calculate", ["d"], operator="divide", left="saving", right="left.value", column="rate"),
                        step("f", "rank", ["e"], column="rate", direction="max")],
                       operands=[operand("QUOTED"), operand("PAID")], column="left.location")
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "South")

    def test_role_bound_join_reconstructs_requested_tuple(self):
        sources, facts = data([1, 2], operands=["CLUB", "VENUE"],
                              roles=[{"object": "Guild", "person": "Mira"},
                                     {"object": "Guild", "location": "Atrium"}])
        facts = [replace(facts[0], operands=("CLUB",)), replace(facts[1], operands=("VENUE",))]
        request = plan([scan("a", "CLUB"), scan("b", "VENUE"),
                        step("c", "join", ["a", "b"], on=[["object", "object"]], cardinality="one_to_one")],
                       operands=[operand("CLUB"), operand("VENUE")], columns=["left.person", "right.location"], shape="tuple_set")
        self.assertEqual(execute_plan(request, facts, sources)["answer"], "Mira | Atrium")

    def test_ambiguous_join_is_not_resolved_by_nearest_or_first_row(self):
        sources, facts = data([1, 2, 3], operands=["A", "B"], roles=[{"object": "Guild"}] * 3)
        facts = [replace(facts[0], operands=("A",)), replace(facts[1], operands=("B",)), replace(facts[2], operands=("B",))]
        request = plan([scan("a", "A"), scan("b", "B"),
                        step("c", "join", ["a", "b"], on=[["object", "object"]], cardinality="one_to_one")],
                       operands=[operand("A"), operand("B")], column="left.object")
        self.assertEqual(execute_plan(request, facts, sources)["reason"], "ambiguous_right_join_operand")

    def test_one_claim_cannot_supply_two_independent_arithmetic_roles(self):
        sources, facts = data([17], operands=["A", "B"])
        request = plan([scan("a", "A"), scan("b", "B"),
                        step("c", "arithmetic", ["a", "b"], operator="subtract", left="value", right="value")],
                       operands=[operand("A", cardinality="scalar"), operand("B", cardinality="scalar")])
        self.assertEqual(execute_plan(request, facts, sources)["reason"], "independent_operands_reuse_one_measurement_claim")

    def test_union_and_intersection_preserve_value_identity(self):
        sources, facts = data([1, 2, 3, 4], operands=["A", "B"], roles=[{"object": name} for name in ("Lute", "Flute", "Flute", "Drum")])
        facts = [replace(fact, operands=("A" if i < 2 else "B",)) for i, fact in enumerate(facts)]
        for operation, expected in (("union", "Lute, Flute, Drum"), ("intersection", "Flute")):
            request = plan([scan("a", "A"), scan("b", "B"), step("c", operation, ["a", "b"], keys=["object"])],
                           operands=[operand("A"), operand("B")], column="object", shape="value_set")
            self.assertEqual(execute_plan(request, facts, sources)["answer"], expected)

    def test_requested_status_union_keeps_ongoing_as_well_as_completed(self):
        sources, facts = data([17, 29])
        ongoing = ground_fact({"source_id": "S2", "quote": facts[1].quote, "actor_span": "I", "predicate": "paid",
                               "status": "ongoing", "kind": "event", "operands": ["O1"], "roles": {},
                               "measurement": {"text": "29 dollars", "value": "29", "unit": "dollars"}}, sources, {"O1"})
        request = plan([scan(), total()], operands=[operand(statuses=["completed", "ongoing"])])
        self.assertEqual(execute_plan(request, [facts[0], ongoing], sources)["answer"], "$46")

    def test_stated_units_cannot_be_erased_to_force_arithmetic(self):
        sources, facts = data([17])
        with self.assertRaisesRegex(ContractError, "cannot_be_erased"):
            ground_fact({"source_id": "S1", "quote": facts[0].quote, "actor_span": "I", "predicate": "paid",
                         "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                         "measurement": {"text": "17 dollars", "value": "17", "unit": "1"}}, sources, {"O1"})

    def test_aliases_require_source_identity_evidence(self):
        sources, facts = data([17], roles=[{"location": "Cedar Shop"}])
        result = execute_plan(plan([scan(), total()]), facts, sources, aliases={("location", "cedar shop"): "Other"})
        self.assertEqual(result["reason"], "alias_source_bindings_required")

    def test_spelled_cardinals_are_grounded_before_unit_conversion(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "I worked for three hours.")
        sources = {"S1": source}
        fact = ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "I", "predicate": "worked",
                            "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                            "measurement": {"text": "three hours", "value": "3", "unit": "hours"}}, sources, {"O1"})
        self.assertEqual(execute_plan(plan([scan(), total()], unit="minutes"), [fact], sources)["answer"], "180 minutes")

    def test_possessive_reference_is_not_the_performing_speaker(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "My brother paid 17 dollars.")
        sources = {"S1": source}
        fact = ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "My", "predicate": "paid",
                            "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                            "measurement": {"text": "17 dollars", "value": "17", "unit": "dollars"}}, sources, {"O1"})
        self.assertFalse(execute_plan(plan([scan(), total()]), [fact], sources)["executable"])

    def test_partial_name_cannot_ground_a_different_entity(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "Isabelle paid 17 dollars at Cedarwood.")
        sources = {"S1": source}
        with self.assertRaisesRegex(ContractError, "word_fragment"):
            ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "I", "predicate": "paid",
                         "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                         "measurement": {"text": "17 dollars", "value": "17", "unit": "dollars"}}, sources, {"O1"})

    def test_stripped_measurement_unit_is_detected_in_authoritative_source(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "I worked for 3 hours.")
        sources = {"S1": source}
        with self.assertRaisesRegex(ContractError, "cannot_be_erased"):
            ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "I", "predicate": "worked",
                         "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                         "measurement": {"text": "3", "value": "3", "unit": "1"}}, sources, {"O1"})

    def test_equal_measurements_in_one_source_remain_distinct_events(self):
        first, second = "I paid 17 dollars at Cedar.", "I paid 17 dollars at Birch."
        source = Source("S1", "s1", "t1", "user", "user", None, first + " " + second)
        sources = {"S1": source}
        facts = [ground_fact({"source_id": "S1", "quote": quote, "actor_span": "I", "predicate": "paid",
                              "status": "completed", "kind": "event", "operands": ["O1"], "roles": {"location": location},
                              "measurement": {"text": "17 dollars", "value": "17", "unit": "dollars"}}, sources, {"O1"})
                 for quote, location in ((first, "Cedar"), (second, "Birch"))]
        self.assertNotEqual(facts[0].event_id, facts[1].event_id)
        self.assertEqual(execute_plan(plan([scan(), total()]), facts, sources)["answer"], "$34")

    def test_approximate_input_cannot_be_certified_as_exact(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "I paid at least 17 dollars.")
        with self.assertRaisesRegex(ContractError, "interval_contract"):
            ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "I", "predicate": "paid",
                         "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                         "measurement": {"text": "17 dollars", "value": "17", "unit": "dollars"}}, {"S1": source}, {"O1"})

    def test_span_only_extraction_uses_source_number_not_model_arithmetic(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "I worked on the mural for 300 minutes.")
        fact = ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "I", "predicate": "worked on",
                            "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                            "measurement": {"text": "300 minutes"}}, {"S1": source}, {"O1"})
        self.assertEqual(fact.raw_value, "300")
        self.assertEqual(fact.quantity.convert("hours"), Decimal(5))

    def test_unique_explicit_measurement_is_not_lost_when_extractor_returns_null(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "I worked on the mural for 3 hours.")
        fact = ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "the mural", "predicate": "worked on",
                            "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                            "measurement": None}, {"S1": source}, {"O1"})
        self.assertEqual(fact.actor, "user")
        self.assertEqual(fact.actor_span, "I")
        self.assertEqual(fact.quantity.convert("minutes"), Decimal(180))

    def test_multiple_source_measures_require_explicit_slot_binding(self):
        source = Source("S1", "s1", "t1", "user", "user", None, "I worked 3 hours and carried 2 kg.")
        fact = ground_fact({"source_id": "S1", "quote": source.text, "actor_span": "I", "predicate": "worked",
                            "status": "completed", "kind": "event", "operands": ["O1"], "roles": {},
                            "measurement": None}, {"S1": source}, {"O1"})
        self.assertIsNone(fact.quantity)
        self.assertFalse(execute_plan(plan([scan(), total()]), [fact], {"S1": source})["executable"])

    def test_raw_quantity_lexing_preserves_known_units_and_rates(self):
        for text, value, unit_name in (("3 hours", "3", "hours"), ("300 minutes", "300", "minutes"),
                                       ("$23", "23", "$"), ("5000g", "5000", "g"),
                                       ("3 dollars per item", "3", "dollars/item")):
            with self.subTest(text=text):
                parsed = parse_measurement_span(text)
                self.assertEqual(parsed, {"text": text, "value": value, "unit": unit_name})
        self.assertEqual(source_measurements("Room 12 is upstairs."), [])


if __name__ == "__main__":
    unittest.main()
