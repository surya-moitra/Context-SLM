"""Real local-model component checks on explicitly supplied fictional evidence.

This is NOT a retrieval benchmark: every example supplies its evidence. Reference
answers are used only after inference. Nothing here is imported by inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace


CASES = (
    ("grouped_total", "At which shop did I spend the most in total on studio supplies?",
     ["I paid $17 at Cedar on studio supplies.", "I paid $38 at Birch on studio supplies.", "I paid $29 at Cedar on studio supplies."], "Cedar"),
    ("remaining", "How many pages of my studio workbook remain unread?",
     ["My studio workbook has 81 pages altogether.", "I've already read 13 pages of that workbook."], "68 pages"),
    ("relative_comparison", "Was my percentage discount on the first studio order higher than on the second?",
     ["For the first studio order the original price was $100.", "For the first studio order I paid $80.",
      "For the second studio order the original price was $60.", "For the second studio order I paid $42."], "No"),
    ("unit_price", "How much did each of my studio mugs cost?",
     ["I paid $56 altogether for the studio mugs.", "There were 7 mugs in that purchase, all at the same price."], "$8"),
    ("typed_rate", "How much did I earn from the studio market sales?",
     ["I sold 9 packets at the studio market.", "Every packet sold for $3."], "$27"),
    ("scalar_mean", "What was my average studio rating, rounded to two decimal places?",
     ["My studio rating in the first term was 3.1.", "My studio rating in the second term was 3.8."], "3.45"),
    ("compound_difference", "By how many seconds was my later studio trial faster?",
     ["My earlier studio trial took 18 minutes and 25 seconds.", "The later studio trial took 16 minutes and 15 seconds."], "130 seconds"),
    ("linked_host", "At which university did I present my studio poster?",
     ["I presented my studio poster at the Lantern Symposium.", "The Lantern Symposium was held at Cedar University; the reception was at Pine College."], "Cedar University"),
    ("calendar_endpoints", "How many months passed between placing and receiving my studio order?",
     ["I placed my studio order on 04 February 2020.", "It arrived on 04 October 2020."], "8 months"),
    ("distinct_weekday", "On how many different days of the week do I attend studio classes?",
     ["I attend studio classes on Monday.", "I attend studio classes on Friday.", "I also attend studio practice on Monday."], "2"),
    ("explicit_missing", "How many years did my studio training take altogether, including the final stage?",
     ["The initial stage of my studio training lasted 3 years.", "I completed the final studio stage, but its duration is unrecorded."], "I do not know."),
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise SystemExit("Choose a new output version")
    chosen = [case for case in CASES if not args.case or case[0] in args.case]
    if args.case and set(args.case) != {case[0] for case in chosen}:
        raise SystemExit("Unknown component case")
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"model_sha256": digest(args.model_path), "source_sha256": {name: digest(Path(__file__).parent / name)
        for name in ("PRAGMOS_typed_claims.py", "PRAGMOS_typed_computation.py", "PRAGMOS_composable_candidate.py",
                     "PRAGMOS_composable_execution.py", "PRAGMOS_grounded_computation.py", "PRAGMOS_probe_typed_computation.py")}}
    from llama_cpp import Llama
    from PRAGMOS_composable_execution import Source
    from PRAGMOS_composable_candidate import run_typed_candidate, promotable, supported_typed_abstention
    llm = Llama(model_path=args.model_path, n_ctx=2048, n_threads=8, n_gpu_layers=40, seed=42, verbose=False)
    context = SimpleNamespace(llm=llm, count_tokens=lambda text: len(llm.tokenize(text.encode(), add_bos=True, special=True)))
    rows = []
    try:
        for name, question, evidence, reference in chosen:
            sources = {f"S{i}": Source(f"S{i}", f"fixture_session_{i}", str(i), "user", "user", f"2028-01-{i:02d}", text)
                       for i, text in enumerate(evidence, 1)}
            started = time.monotonic()
            result = run_typed_candidate(context, question, sources, n_ctx=2048,
                                         answer_contract={"extended_contracts": True, "preserve_slots_v3": True})
            accepted = promotable(result) or supported_typed_abstention(result)
            row = {"case": name, "question": question, "supplied_evidence": evidence,
                   "prediction": result.get("answer") if accepted else None, "reference": reference,
                   "exact_match": accepted and result["answer"] == reference,
                   "elapsed_seconds": time.monotonic() - started, "candidate": result}
            rows.append(row)
            with path.with_suffix(".partial.json").open("w") as stream:
                json.dump({"complete": False, "identity": identity, "rows": rows}, stream, indent=2)
            print(json.dumps({**{key: row[key] for key in ("case", "prediction", "reference", "exact_match")},
                              "reason": result.get("reason")}), flush=True)
    finally:
        llm.close()
        report = {"scope": "fictional_supplied_evidence_component_check_not_retrieval_or_benchmark_accuracy",
                  "identity": identity, "n_ctx": 2048, "complete": len(rows) == len(chosen), "rows": rows}
        with path.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
