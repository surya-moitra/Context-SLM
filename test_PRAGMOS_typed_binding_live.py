"""Optional real-local-model checks on fictional evidence, not benchmark cases."""

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest

from PRAGMOS_composable_candidate import _call
from PRAGMOS_grounded_computation import encoded
from PRAGMOS_typed_binding import bind_operand_properties, bind_scalar_operands, refine_request, select_relation_claims
from PRAGMOS_typed_claims import source_windows
from PRAGMOS_composable_execution import Source
from PRAGMOS_typed_computation import execute_typed_plan
from test_PRAGMOS_typed_computation import bind, request, selection


class LocalBindingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = os.environ.get("PRAGMOS_TYPED_BINDING_LIVE_MODEL")
        if not model:
            raise unittest.SkipTest("Set PRAGMOS_TYPED_BINDING_LIVE_MODEL for local inference checks")
        from llama_cpp import Llama
        cls.llm = Llama(model_path=model, n_ctx=2048, n_threads=8, n_gpu_layers=40, seed=42, verbose=False)
        cls.context = SimpleNamespace(llm=cls.llm, count_tokens=lambda text: len(cls.llm.tokenize(text.encode(), add_bos=True, special=True)))

    @classmethod
    def tearDownClass(cls):
        cls.llm.close()

    def setUp(self):
        self.audit = {"planner_calls": 0, "extractor_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.selections = []

    def call(self, prompt, stage, schema, reserve):
        result = _call(self.context, prompt, stage, 2048, self.audit, schema=schema, max_tokens=reserve)
        self.selections.append(result)
        return result

    def test_temporal_growth_selects_net_change(self):
        base = replace(request("subtract", grouped=True, rank="max", projection="entity"),
                       left_operand="initial height", right_operand="final height")
        result = refine_request(self.call, "Which of my gardens increased more in height during May 2023?", base,
                                [{"text": "My Cedar garden was measured on 2 May 2023 and 30 May 2023."}])
        self.assertEqual(result.operand_order, (("left", "latest"), ("right", "earliest")), encoded(self.selections))

    def test_comparison_phrase_is_not_biased_by_source_prices(self):
        base = replace(request("relative_difference", grouped=True, comparison="less"),
                       key_description="bigger than", left_operand="reference baseline", right_operand="actual value")
        result = refine_request(self.call, "Was my relative reduction on the first order bigger than on the second?", base, [])
        self.assertEqual(result.comparison, "greater", encoded(self.selections))

    def test_source_property_and_group_order_selection(self):
        claims, windows, sources = bind(["The first listed price was $150.", "The first payment was $120.",
                                        "The second listed price was $200.", "The second payment was $170."], [
            selection(actor=None, binding="event_property", event=group, entity=group, key=group, attribute=prop,
                      numbers=[0], quantity_kind="measure")
            for group, prop in (("first", "listed price"), ("first", "payment"), ("second", "listed price"), ("second", "payment"))])
        base = replace(request("relative_difference", grouped=True, comparison="greater"), property_binding=True,
                       left_operand="reference baseline", right_operand="actual value")
        bound, _ = bind_operand_properties(self.call, "Was the percentage reduction on the first order greater than on the second?",
                                           base, claims, windows, sources)
        self.assertEqual(execute_typed_plan(base, bound, windows, sources)["answer"], "Yes", encoded(self.selections))

    def test_duration_operand_order(self):
        claims, windows, sources = bind(["My earlier practice took 38 seconds.", "The later practice took 31 seconds."], [
            selection(actor="My", entity="earlier practice", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="later practice", entity="later practice", numbers=[0], quantity_kind="measure")])
        base = replace(request("subtract"), value_binding=True)
        bound, _ = bind_scalar_operands(self.call, "By how many seconds was my later practice shorter?", base, claims, windows, sources)
        self.assertEqual(execute_typed_plan(base, bound, windows, sources)["answer"], "7 seconds", encoded(self.selections))

    def test_relation_extraction_keeps_indirect_venue(self):
        source = Source("S", "s", "t", "user", "user", None, "I displayed the studio poster at Lantern Forum.")
        window = source_windows({"S": source})[0]
        base = replace(request("lookup", projection="entity", key_role="location"),
                       key_description="university", source_addressed=True)
        payload = select_relation_claims(self.call, "At which university did I display the studio poster?", base, window, source, {})
        self.assertTrue(any("Lantern Forum" in row["key"] for row in payload["facts"]), encoded(self.selections))

    def test_faster_binds_to_measured_elapsed_duration(self):
        claims, windows, sources = bind(["My earlier practice took 38 seconds.", "The later practice took 31 seconds."], [
            selection(actor="My", entity="earlier practice", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="later practice", entity="later practice", numbers=[0], quantity_kind="measure")])
        base = replace(request("subtract", question="In seconds?", output_unit="seconds"), value_binding=True)
        bound, _ = bind_scalar_operands(self.call, "By how many seconds was my later practice faster?", base, claims, windows, sources)
        self.assertEqual(execute_typed_plan(base, bound, windows, sources)["answer"], "7 seconds", encoded(self.selections))

    def test_related_count_owner_is_not_the_counted_object(self):
        base = request("count", projection="entity", grouped=True, rank="max")
        try:
            refined = refine_request(self.call, "How many children did my sisters welcome during July 2023?", base, [])
        except Exception as exc:
            self.fail(str(exc) + ": " + encoded(self.selections))
        self.assertEqual((refined.projection, refined.rank, refined.actor_scope), ("value", "none", "related"), encoded(self.selections))

    def test_duration_increase_preserves_its_sign(self):
        claims, windows, sources = bind(["My earlier practice took 38 seconds.", "The later practice took 31 seconds."], [
            selection(actor="My", entity="earlier practice", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="later practice", entity="later practice", numbers=[0], quantity_kind="measure")])
        base = replace(request("subtract", question="In seconds?", output_unit="seconds"), value_binding=True)
        bound, _ = bind_scalar_operands(self.call, "What was the signed increase in the later practice's time taken, in seconds?", base, claims, windows, sources)
        self.assertEqual(execute_typed_plan(base, bound, windows, sources)["answer"], "-7 seconds", encoded(self.selections))

    def test_duration_order_is_not_retrieval_order(self):
        claims, windows, sources = bind(["My earlier practice took 38 seconds.", "The later practice took 31 seconds."], [
            selection(actor="My", entity="earlier practice", numbers=[0], quantity_kind="measure"),
            selection(actor=None, binding="event_property", event="later practice", entity="later practice", numbers=[0], quantity_kind="measure")])
        base = replace(request("subtract", question="In seconds?", output_unit="seconds"), value_binding=True)
        bound, _ = bind_scalar_operands(self.call, "By how many seconds was my later practice shorter?", base,
                                        list(reversed(claims)), windows, sources)
        self.assertEqual(execute_typed_plan(base, bound, windows, sources)["answer"], "7 seconds", encoded(self.selections))


if __name__ == "__main__":
    unittest.main()
