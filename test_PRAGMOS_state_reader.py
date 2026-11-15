"""Independent speech-act/relevance fixtures and assertion coverage regressions."""

from dataclasses import replace
import unittest

from PRAGMOS_composable_execution import Source
from PRAGMOS_state_contracts import bind_answer_contract
from PRAGMOS_state_owners import OwnerIndex
from PRAGMOS_state_reader import _owner, read_source_window
from PRAGMOS_typed_claims import SELF, actor_choices, numeric_slots
from test_PRAGMOS_state_contracts import contract_payloads, source_fact


def reader_responses(source, rows, *, omitted=None, dispositions=None):
    fields = ("assertion", "value")
    replies = [{"disposition": "related"}]
    actors = list(actor_choices(source.text))
    if (any(len(a.split()) > 1 and a.split()[0].casefold() in SELF for a in actors)
            and any(a[:1].isupper() and a.split()[0].casefold() not in SELF for a in actors)):
        replies.append({"quote": None})
    replies.append({"facts": [{key: row[key] for key in fields} for row in rows], "unresolved": False})
    categories = {"snapshot": "observation", "reference": "archival_mention", "correction": "corrected_report",
                  "add": "membership_addition", "remove": "membership_removal", "denied": "negated_fact",
                  "context": "background", "unresolved": "uncertain"}
    for i, row in enumerate(rows):
        assertion = __import__('PRAGMOS_state_binding').assertion_spans(source.text)[int(row["assertion"])]
        replies.append({"predicate": row["predicate"] if row["predicate"] in assertion else assertion,
            "property": row["property"] if row["property"] and row["property"] in assertion else None,
            "entity": row["entity"] if row["entity"] and row["entity"] in assertion else None})
        replies.append({"assertion_kind": "future_plan" if row["modality"] == "planned" else categories[row["kind"]]})
        slots = numeric_slots(row["value"])
        if not any(s.raw_unit for s in slots):
            replies.append({"value_type": row["value_type"]})
        replies.append({"value_role": row["value_role"]})
        if any(other["value"] != row["value"] and other["assertion"] == row["assertion"] for other in rows) and row["kind"] not in {"reference", "context"}:
            replies.append({"previous": row["previous"]})
        match = dispositions[i] if dispositions else row["operand"] or "irrelevant"
        replies.append({"operand": match})
        if match == "irrelevant":
            continue
        assertion = __import__('PRAGMOS_state_binding').assertion_spans(source.text)[int(row["assertion"])]
        candidates = list(actor_choices(assertion)) or list(actor_choices(source.text))
        replies.append({"owner_index": str(candidates.index(row["actor"]))})
        if row["actor"].casefold() not in SELF:
            replies.append({"owner_kind": row["binding"]})
        if row["dimension"] == "count" and row["value_type"] == "quantity":
            replies.append({"counted_entity": row["counted_entity"]})
    if omitted:
        replies.append(omitted)
    return replies


class ReaderTests(unittest.TestCase):
    def test_unowned_location_is_not_reinterpreted_as_an_object_rule(self):
        text = "The spare leads are on the cupboard shelf."
        source = Source("S1", "s1", "1", "user", "user", "2025-01-01", text)
        calls = []
        def ask(*args):
            calls.append(args)
            self.assertEqual(len(calls), 1, "Unbound location must not open the rule fallback")
            return {"owner_index": None}
        owners = OwnerIndex()
        result = _owner(ask, {"value": "cupboard shelf"}, text, source, text, owners)
        self.assertEqual(result, (None, None, None, None))
        self.assertEqual(owners.rule_witnesses, [])

    def test_explicitly_permitted_quantity_rule_retains_a_source_witness(self):
        text = "A standard reward night costs 2800 points."
        source = Source("S2", "s2", "2", "user", "user", "2025-01-01", text)
        replies = iter([{"owner_index": None}, {"scope": "object_rule"}])
        owners = OwnerIndex()
        key, kind, witness, actor = _owner(lambda *args: next(replies), {"value": "2800 points"},
                                         text, source, text, owners, allow_rule=True)
        self.assertTrue(key.startswith("rule:"))
        self.assertEqual((kind, witness, actor), ("rule", text, None))
        self.assertEqual(owners.rule_witnesses[0]["source_id"], "S2")

    def run_rows(self, text, rows, *, omitted=None, dispositions=None):
        source = Source("S1", "s1", "1", "user", "user", "2025-01-01", text)
        contract = bind_answer_contract("What is my balance?", *contract_payloads("What is my balance?"))
        answers = iter(reader_responses(source, rows, omitted=omitted, dispositions=dispositions))
        prompts = []
        def ask(prompt, *args):
            prompts.append(prompt)
            return next(answers)
        report = {}
        result = read_source_window(ask, contract, source, 0, text, 1, OwnerIndex(), report)
        return result, report, prompts

    def test_second_assertion_uses_its_own_literal_schema(self):
        text = "My balance was $20. My balance is $35."
        result, _, _ = self.run_rows(text, [source_fact("$20", predicate="was"), source_fact("$35", assertion="1")])
        self.assertEqual([f.value for f in result], ["$20", "$35"])

    def test_source_status_prompt_contains_no_question_or_answer_contract(self):
        _, _, prompts = self.run_rows("My balance is $20.", [source_fact("$20")])
        status = [p for p in prompts if p.startswith("Classify this source assertion")]
        self.assertEqual(len(status), 1)
        self.assertNotIn("What is my balance?", status[0])
        self.assertNotIn("Contract:", status[0])

    def test_uncertain_unrelated_fact_does_not_abort_a_valid_assertion(self):
        rows = [source_fact("$20"), source_fact("$40", assertion="1", kind="unresolved")]
        result, report, _ = self.run_rows("My balance is $20. My deposit might be $40.", rows, dispositions=["P1", "irrelevant"])
        self.assertEqual([f.value for f in result], ["$20"])
        self.assertFalse(report.get("assertion_uncertainty"))

    def test_uncertain_required_fact_is_retained_as_a_blocking_obligation(self):
        result, report, _ = self.run_rows("My balance is $20.", [source_fact("$20", kind="unresolved")])
        self.assertEqual(result, [])
        self.assertEqual(report["assertion_uncertainty"][0]["operands"], ["P1"])

    def test_omitted_potentially_relevant_assertion_is_not_absence(self):
        result, report, _ = self.run_rows("My balance is $20. I corrected the balance later.", [source_fact("$20")], omitted={"1": "unresolved"})
        self.assertEqual(len(result), 1)
        self.assertEqual(report["assertion_uncertainty"][0]["reason"], "unextracted_assertion_requires_review")


if __name__ == "__main__":
    unittest.main()
