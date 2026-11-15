"""Contract/extraction counterexamples; fixture model choices are not accuracy."""

from dataclasses import replace
import unittest

from PRAGMOS_composable_execution import ContractError, Source
from PRAGMOS_state_contracts import bind_answer_contract, plan_answer_contract
from PRAGMOS_state_binding import bind_source_relations, ordered_history, select_history, choose_executable_query


def contract_payloads(question, property_ref="balance", *, projection="measurement", input_type="quantity",
                      dimension="currency", family="read", shape="one", selector="latest", **scope_changes):
    output = {"request_ref": question, "projection": projection, "input_type": input_type,
              "dimension": dimension, "counted_entity": None, "shape": shape, "operation_family": family}
    scope = {"subject_ref": None, "owner_scope": "speaker", "target_ref": None, "relation_ref": property_ref,
             "qualifiers": [], "modality": "actual", "time_selector": selector, "time_ref": None,
             "operand_refs": [property_ref]}
    scope.update(scope_changes)
    return [output, scope]


def planner_responses(payloads):
    output, scope = payloads
    replies = [{"projection": output["projection"]}, {"family": output["operation_family"]}]
    if output["projection"] not in {"boolean", "direction"}:
        replies.append({"shape": output["shape"]})
    if output["projection"] == "boolean":
        replies.append({"input_type": output["input_type"]})
    if output["projection"] == "cardinality":
        replies.append({"counted_entity": output["counted_entity"]})
    replies += [{key: scope[key] for key in ("subject_ref", "owner_scope")},
                {"operand_refs": scope["operand_refs"]},
                {key: scope[key] for key in ("target_ref", "qualifiers")},
                {key: scope[key] for key in ("time_selector", "time_ref", "modality")}]
    return replies


def source_fact(value, *, assertion="0", actor="My", predicate="is", property_ref="balance", **changes):
    row = {"assertion": assertion, "actor": actor, "binding": "speaker", "predicate": predicate,
           "property": property_ref, "entity": None, "value": value, "value_role": "measurement",
           "value_type": "quantity", "dimension": "currency", "operand": "P1", "counted_entity": None,
           "kind": "snapshot", "modality": "actual", "qualifiers": [], "time_ref": None,
           "previous": None, "transition": None}
    row.update(changes)
    return row


def relation_responses(rows):
    fields = ("assertion", "actor", "predicate", "property", "entity", "value", "counted_entity")
    responses = [{"facts": [{key: row[key] for key in fields} for row in rows], "unresolved": False}]
    for row in rows:
        responses.append({"meaning": "2" if row["kind"] == "reference" else "3" if row["modality"] == "planned" else "1",
                          "value_role": row["value_role"], **{key: row[key] for key in (
                              "time_ref", "qualifiers", "previous", "transition")}})
    return responses


def source(text, day=1):
    return Source("S" + str(day), "session" + str(day), str(day), "user", "user", f"2025-01-{day:02d}", text)


def bind_rows(question, payloads, text, rows, report=None):
    contract = bind_answer_contract(question, *payloads)
    item = source(text)
    return bind_source_relations({"facts": rows, "unresolved": False}, contract, item, 0, text, 1,
                                 report if report is not None else {})


class AnswerContractTests(unittest.TestCase):
    def test_complete_relation_is_not_limited_to_four_words(self):
        q = "What balance is available on my account after all pending payments?"
        relation = "balance is available on my account after all pending payments"
        result = bind_answer_contract(q, *contract_payloads(q, relation))
        self.assertEqual(result.relation_ref, relation)
        self.assertEqual(result.required_operands[0].request_ref, relation)

    def test_query_planning_never_reads_source_values(self):
        q = "How long have I followed my routine?"
        payloads = planner_responses(contract_payloads(q, "routine", projection="duration", dimension="duration"))
        replies = iter(payloads)
        prompts = []
        def ask(prompt, *args):
            prompts.append(prompt)
            return next(replies)
        result = plan_answer_contract(ask, q)
        self.assertEqual(result.projection, "duration")
        self.assertEqual(len(prompts), len(payloads))
        self.assertTrue(all(q in prompt for prompt in prompts))

    def test_invented_scope_or_operand_cannot_be_bound(self):
        q = "What is my balance?"
        for field, value in (("target_ref", "invented account"), ("operand_refs", ["invented balance"]),
                             ("qualifiers", ["invented restriction"])):
            with self.subTest(field=field):
                payloads = contract_payloads(q)
                payloads[1][field] = value
                with self.assertRaisesRegex(ContractError, "span_not_literal"):
                    bind_answer_contract(q, *payloads)

    def test_duration_cannot_be_declared_a_text_input(self):
        q = "How long is my walk?"
        with self.assertRaisesRegex(ContractError, "requires_time_quantity"):
            bind_answer_contract(q, *contract_payloads(q, "walk", projection="duration", input_type="text", dimension="none"))

    def test_location_cannot_be_declared_a_quantity_input(self):
        q = "Where is my instrument?"
        with self.assertRaisesRegex(ContractError, "text_projection"):
            bind_answer_contract(q, *contract_payloads(q, "instrument", projection="location"))

    def test_direction_is_a_text_answer_even_when_question_mentions_two_states(self):
        q = "Did my latest change increase or decrease my limit?"
        result = bind_answer_contract(q, *contract_payloads(q, "limit", projection="direction",
                                      dimension="count", family="compare", shape="boolean"))
        self.assertEqual(result.shape, "one")
        self.assertEqual(result.operation_family, "compare")

    def test_all_distinct_operands_are_retained(self):
        q = "How many nights can my points buy at the nightly rate?"
        payloads = contract_payloads(q, "points", dimension="count", family="calculate",
                                    operand_refs=["points", "nightly rate"])
        result = bind_answer_contract(q, *payloads)
        self.assertEqual([item.relation_id for item in result.required_operands], ["P1", "P2"])


class SourceRelationTests(unittest.TestCase):
    def test_numeral_inside_property_name_is_not_a_measurement_value(self):
        q, text = "What is my best 5K time?", "My 5K time is 24 minutes."
        rows = bind_rows(q, contract_payloads(q, "5K time", projection="duration", dimension="duration"), text,
                         [source_fact("24 minutes", property_ref="5K time", value_role="duration", dimension="duration")])
        self.assertEqual(rows[0].relation.property_ref, "5K time")

    def test_location_slot_rejects_action_even_when_both_are_literal(self):
        q, text = "Where did I repair my instrument?", "I adjusted the bridge at Reed Workshop."
        payloads = contract_payloads(q, "repair", projection="location", input_type="text", dimension="none")
        row = source_fact("adjusted the bridge", actor="I", predicate="adjusted", property_ref="bridge",
                          value_role="attribute", value_type="text", dimension="none")
        with self.assertRaisesRegex(ContractError, "answer_role_mismatch"):
            bind_rows(q, payloads, text, [row])

    def test_duration_slot_retains_elapsed_quantity_instead_of_activity_name(self):
        q, text = "How long have I followed my routine?", "I followed my routine for 7 weeks."
        payloads = contract_payloads(q, "routine", projection="duration", dimension="duration")
        row = source_fact("7 weeks", actor="I", predicate="followed", property_ref="routine",
                          value_role="duration", dimension="duration")
        result = bind_rows(q, payloads, text, [row])[0]
        self.assertEqual(result.value, "7 weeks")
        self.assertEqual(result.relation.value.dimension, "duration")
        with self.assertRaises(ContractError):
            bind_rows(q, payloads, text, [dict(row, value="routine")])

    def test_same_unit_does_not_merge_balance_and_requirement(self):
        q, text = "What is my balance?", "My balance is $90. My threshold is $30."
        a, b = source_fact("$90"), source_fact("$30", assertion="1", property_ref="threshold")
        rows = bind_rows(q, contract_payloads(q), text, [a, b])
        with self.assertRaisesRegex(ContractError, "crosses_properties"):
            ordered_history(rows)

    def test_same_counted_noun_does_not_merge_different_source_predicates(self):
        q, text = "How many credits do I need?", "I need 75 credits. I have 110 credits."
        payloads = contract_payloads(q, "credits", projection="cardinality", dimension="count")
        rows = bind_rows(q, payloads, text, [
            source_fact("75 credits", actor="I", predicate="need", property_ref="credits", dimension="count", value_role="cardinality"),
            source_fact("110 credits", assertion="1", actor="I", predicate="have", property_ref="credits", dimension="count", value_role="cardinality")])
        with self.assertRaisesRegex(ContractError, "source_predicates_require_binding"):
            ordered_history(rows)

    def test_unrelated_property_remains_in_audit_without_entering_history(self):
        q, text, audit = "What is my balance?", "My balance is $90. My threshold is $30.", {}
        rows = bind_rows(q, contract_payloads(q), text, [source_fact("$90"),
                         source_fact("$30", assertion="1", property_ref="threshold", operand=None)], audit)
        self.assertEqual([row.value for row in rows], ["$90"])
        self.assertEqual(len(audit["source_relations"]), 2)
        self.assertEqual(audit["source_relations"][1]["relation"]["property_ref"], "threshold")

    def test_record_with_only_numbers_is_not_an_unresolved_pronoun(self):
        q, text = "What is my current record?", "My record is 7-3."
        payloads = contract_payloads(q, "record", projection="record", input_type="record", dimension="none")
        row = source_fact("7-3", property_ref="record", value_role="record", value_type="record", dimension="none")
        result = bind_rows(q, payloads, text, [row])[0]
        self.assertEqual(result.value, "7-3")
        self.assertEqual(result.relation.value.kind, "record")

    def test_range_and_compound_duration_are_preserved(self):
        for value in ("about 12-16 hours", "18 minutes 14 seconds"):
            with self.subTest(value=value):
                q, text = "What is my duration?", "My duration is " + value + "."
                rows = bind_rows(q, contract_payloads(q, "duration", projection="duration", dimension="duration"),
                                 text, [source_fact(value, property_ref="duration", value_role="duration", dimension="duration")])
                self.assertEqual(rows[0].relation.value.raw, value)

    def test_list_order_alone_does_not_create_a_state_transition(self):
        q, text = "What is my balance?", "My balance was $20 and is now $35."
        rows = bind_rows(q, contract_payloads(q), text, [source_fact("$35"), source_fact("$20")])
        self.assertTrue(all(row.previous is None for row in rows))
        with self.assertRaisesRegex(ContractError, "explicit_order"):
            select_history(rows, "latest")

    def test_explicit_transition_orders_values_without_using_array_order(self):
        q, text = "What is my balance?", "My balance was $20 and is now $35."
        rows = bind_rows(q, contract_payloads(q), text, [source_fact("$35", previous="$20", transition=text), source_fact("$20")])
        self.assertEqual(select_history(rows, "latest")[0].value, "$35")
        self.assertEqual(rows[0].transition_quote, text)

    def test_planned_goal_and_archived_reference_do_not_change_actual_history_identity(self):
        from PRAGMOS_state_binding import bind_query, execute_query
        q, text = "What is my balance?", "My balance is $20. My target balance will be $35. My receipt recorded $10."
        rows = bind_rows(q, contract_payloads(q), text, [
            source_fact("$20"),
            source_fact("$35", assertion="1", predicate="will be", property_ref="target balance", modality="planned"),
            source_fact("$10", assertion="2", predicate="recorded", property_ref="receipt", kind="reference")])
        plan = bind_query({"operation": "read", "left": [row.fact_id for row in rows], "right": [],
                          "selectors": ["latest"], "marker": None, "exclusions": [], "modality": "actual",
                          "whole_units": False, "request_quote": q}, q, rows)
        self.assertEqual(execute_query(plan, rows)["answer"], "$20")

    def test_fabricated_transition_and_borrowed_predicate_are_rejected(self):
        q, text = "What is my balance?", "My balance is $20. My threshold exceeds $10."
        for changes in ({"previous": "$10", "transition": "invented"}, {"predicate": "exceeds"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ContractError):
                    bind_rows(q, contract_payloads(q), text, [source_fact("$20", **changes)])

    def test_units_and_counted_entity_are_checked_independently_of_model_binding(self):
        q, text = "How many nights can I redeem?", "My balance is 8100 points."
        payloads = contract_payloads(q, "nights", projection="cardinality", dimension="count")
        payloads[0]["counted_entity"] = "nights"
        row = source_fact("8100", value_role="cardinality", dimension="count", counted_entity="points")
        with self.assertRaisesRegex(ContractError, "counted_entity"):
            bind_rows(q, payloads, text, [row])

    def test_bare_numbers_cannot_claim_duration_without_a_source_unit(self):
        q, text = "How long is my duration?", "My duration is 17."
        with self.assertRaisesRegex(ContractError, "unit_not_bound"):
            bind_rows(q, contract_payloads(q, "duration", projection="duration", dimension="duration"), text,
                      [source_fact("17", property_ref="duration", value_role="duration", dimension="duration")])

    def test_bound_read_cannot_be_changed_to_minimum_by_result_menu(self):
        q = "What is my balance?"
        contract = bind_answer_contract(q, *contract_payloads(q))
        a = bind_rows(q, contract_payloads(q), "My balance is $20.", [source_fact("$20")])[0]
        b = replace(a, fact_id="F2", timestamp="2025-01-02", value="$35")
        def forbidden(*args):
            self.fail("An answer-bearing operation menu must not override the read contract")
        plan = choose_executable_query(forbidden, q, [a, b], answer_contract=contract)
        self.assertEqual(plan["operation"], "read")


if __name__ == "__main__":
    unittest.main()
