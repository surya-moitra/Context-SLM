#!/usr/bin/env python3
"""Compare synthetic V2 semantic difficulty with the full LongMemEval corpus."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
STOPWORDS = {
    "a", "about", "after", "an", "and", "are", "as", "at", "be", "been",
    "before", "being", "but", "by", "can", "could", "current", "did", "do",
    "does", "earlier", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "last", "later", "me", "mine", "my", "now", "of", "on",
    "or", "our", "ours", "over", "she", "should", "that", "than", "the",
    "their", "theirs", "them", "then", "these", "they", "this", "those",
    "to", "under", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "without", "would", "you", "your",
    "yours",
}
ANSWER_BOILERPLATE = {
    "answer", "based", "confirmed", "concerning", "details", "earlier",
    "exact", "excluded", "facts", "information", "mention", "mentioned",
    "names", "other", "people", "preserved", "provided", "quantities",
    "recorded", "supported", "tentative", "units", "values",
}


def words(value: object, remove_stopwords: bool = True) -> list[str]:
    tokens = [token.lower() for token in WORD_RE.findall(str(value or ""))]
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def normalized_text(value: object) -> str:
    return " ".join(words(value, remove_stopwords=False))


def mean(values: list[float | int]) -> float:
    return statistics.mean(values) if values else 0.0


def relative_delta(candidate: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if candidate == 0 else float("inf")
    return (candidate - reference) / reference


def query_overlap(question: str, text: str) -> float:
    query = set(words(question))
    evidence = set(words(text))
    return len(query & evidence) / len(query) if query else 0.0


def answer_token_coverage(answer: str, evidence: str) -> float:
    answer_tokens = [
        token for token in words(answer) if token not in ANSWER_BOILERPLATE
    ]
    evidence_tokens = set(words(evidence, remove_stopwords=False))
    if not answer_tokens:
        return 0.0
    return sum(token in evidence_tokens for token in answer_tokens) / len(
        answer_tokens
    )


def load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return data


def official_relevance_map(
    official_records: list[dict], oracle_records: list[dict]
) -> dict[str, set[tuple[str, int]]]:
    official_by_id = {
        str(record.get("question_id")): record for record in official_records
    }
    result = {}
    for oracle in oracle_records:
        question_id = str(oracle.get("question_id"))
        official = official_by_id.get(question_id)
        if official is None:
            raise ValueError(f"Oracle question missing from official data: {question_id}")
        official_sessions = dict(
            zip(official["haystack_session_ids"], official["haystack_sessions"])
        )
        locations = set()
        for session_id, oracle_session in zip(
            oracle["haystack_session_ids"], oracle["haystack_sessions"]
        ):
            official_session = official_sessions.get(session_id)
            if official_session is None:
                raise ValueError(
                    f"Oracle session missing from official data: {question_id} {session_id}"
                )
            for turn_index, message in enumerate(oracle_session):
                if turn_index >= len(official_session):
                    raise ValueError(
                        f"Oracle turn index is invalid: {question_id} {session_id}"
                    )
                if normalized_text(message.get("content")) != normalized_text(
                    official_session[turn_index].get("content")
                ):
                    raise ValueError(
                        f"Oracle turn differs from official data: {question_id} "
                        f"{session_id}:{turn_index}"
                    )
                if message.get("has_answer") is True:
                    locations.add((session_id, turn_index))
        result[question_id] = locations
    if set(result) != set(official_by_id):
        raise ValueError("Official data and oracle question IDs differ")
    return result


def synthetic_relevance_map(audit_path: Path) -> dict[str, set[tuple[str, int]]]:
    result = {}
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            audit = json.loads(line)
            result[str(audit["question_id"])] = {
                (str(evidence["session_id"]), int(evidence["turn_index"]))
                for evidence in audit.get("evidence") or []
            }
    return result


def summarize_rows(rows: list[dict]) -> dict:
    non_abstention = [row for row in rows if not row["abstention"]]
    with_evidence = [row for row in rows if row["relevant_turn_count"]]
    return {
        "records": len(rows),
        "relevant_turn_count_mean": mean(
            [row["relevant_turn_count"] for row in rows]
        ),
        "relevant_session_count_mean": mean(
            [row["relevant_session_count"] for row in rows]
        ),
        "zero_relevant_record_rate": mean(
            [row["relevant_turn_count"] == 0 for row in rows]
        ),
        "best_relevant_query_overlap_mean": mean(
            [row["best_relevant_query_overlap"] for row in rows]
        ),
        "best_distractor_query_overlap_mean": mean(
            [row["best_distractor_query_overlap"] for row in rows]
        ),
        "distractors_at_or_above_best_relevant_mean": mean(
            [row["distractors_at_or_above_best_relevant"] for row in rows]
        ),
        "relevant_is_lexical_top1_rate": mean(
            [row["relevant_is_lexical_top1"] for row in rows]
        ),
        "answer_exact_in_evidence_rate_non_abstention": mean(
            [row["answer_exact_in_evidence"] for row in non_abstention]
        ),
        "answer_token_coverage_mean_non_abstention": mean(
            [row["answer_token_coverage"] for row in non_abstention]
        ),
        "evidence_sentence_prefix_unique_rate_mean": mean(
            [row["sentence_prefix_unique_rate"] for row in with_evidence]
        ),
        "evidence_type_token_ratio_mean": mean(
            [row["evidence_type_token_ratio"] for row in with_evidence]
        ),
    }


def dataset_semantics(
    records: list[dict],
    relevance: dict[str, set[tuple[str, int]]],
) -> dict:
    rows = []
    grouped = defaultdict(list)
    for record in records:
        question_id = str(record.get("question_id"))
        locations = relevance.get(question_id)
        if locations is None:
            raise ValueError(f"Missing relevance data for {question_id}")
        turns = [
            (session_id, turn_index, message)
            for session_id, session in zip(
                record["haystack_session_ids"], record["haystack_sessions"]
            )
            for turn_index, message in enumerate(session)
        ]
        relevant_texts = [
            str(message.get("content") or "")
            for session_id, turn_index, message in turns
            if (session_id, turn_index) in locations
        ]
        distractor_texts = [
            str(message.get("content") or "")
            for session_id, turn_index, message in turns
            if (session_id, turn_index) not in locations
        ]
        relevant_scores = [
            query_overlap(record["question"], text) for text in relevant_texts
        ]
        distractor_scores = [
            query_overlap(record["question"], text) for text in distractor_texts
        ]
        best_relevant = max(relevant_scores, default=0.0)
        evidence_text = " ".join(relevant_texts)
        sentences = [
            sentence.strip().lower()
            for text in relevant_texts
            for sentence in SENTENCE_RE.findall(text)
            if sentence.strip()
        ]
        prefixes = [
            " ".join(words(sentence, remove_stopwords=False)[:4])
            for sentence in sentences
        ]
        evidence_tokens = words(evidence_text)
        normalized_answer = normalized_text(record.get("answer"))
        row = {
            "question_id": question_id,
            "question_type": str(record.get("question_type")),
            "abstention": "_abs" in question_id,
            "relevant_turn_count": len(relevant_texts),
            "relevant_session_count": len({session_id for session_id, _ in locations}),
            "best_relevant_query_overlap": best_relevant,
            "best_distractor_query_overlap": max(distractor_scores, default=0.0),
            "distractors_at_or_above_best_relevant": (
                sum(score >= best_relevant for score in distractor_scores)
                if relevant_texts
                else len(distractor_scores)
            ),
            "relevant_is_lexical_top1": int(
                bool(relevant_texts)
                and not any(score >= best_relevant for score in distractor_scores)
            ),
            "answer_exact_in_evidence": int(
                bool(normalized_answer)
                and normalized_answer in normalized_text(evidence_text)
            ),
            "answer_token_coverage": answer_token_coverage(
                str(record.get("answer") or ""), evidence_text
            ),
            "sentence_prefix_unique_rate": (
                len(set(prefixes)) / len(prefixes) if prefixes else 0.0
            ),
            "evidence_type_token_ratio": (
                len(set(evidence_tokens)) / len(evidence_tokens)
                if evidence_tokens
                else 0.0
            ),
        }
        rows.append(row)
        grouped[row["question_type"]].append(row)
    return {
        "overall": summarize_rows(rows),
        "by_question_type": {
            question_type: summarize_rows(values)
            for question_type, values in sorted(grouped.items())
        },
    }


def compare(official: dict, synthetic: dict) -> dict:
    def delta(metric: str) -> dict:
        reference = official["overall"][metric]
        candidate = synthetic["overall"][metric]
        return {
            "official": reference,
            "synthetic": candidate,
            "absolute_delta": candidate - reference,
            "relative_delta": relative_delta(candidate, reference),
        }

    metrics = {
        metric: delta(metric)
        for metric in [
            "relevant_turn_count_mean",
            "relevant_session_count_mean",
            "zero_relevant_record_rate",
            "best_relevant_query_overlap_mean",
            "best_distractor_query_overlap_mean",
            "distractors_at_or_above_best_relevant_mean",
            "relevant_is_lexical_top1_rate",
            "answer_exact_in_evidence_rate_non_abstention",
            "answer_token_coverage_mean_non_abstention",
            "evidence_sentence_prefix_unique_rate_mean",
            "evidence_type_token_ratio_mean",
        ]
    }
    type_turn_checks = {
        question_type: abs(
            relative_delta(
                synthetic["by_question_type"][question_type][
                    "relevant_turn_count_mean"
                ],
                official["by_question_type"][question_type][
                    "relevant_turn_count_mean"
                ],
            )
        ) <= 0.05
        for question_type in official["by_question_type"]
    }
    checks = {
        "relevant_turn_count_mean_within_5pct": abs(
            metrics["relevant_turn_count_mean"]["relative_delta"]
        ) <= 0.05,
        "relevant_session_count_mean_within_10pct": abs(
            metrics["relevant_session_count_mean"]["relative_delta"]
        ) <= 0.10,
        "zero_relevant_record_rate_exact": metrics[
            "zero_relevant_record_rate"
        ]["absolute_delta"] == 0,
        "best_relevant_overlap_mean_within_20pct": abs(
            metrics["best_relevant_query_overlap_mean"]["relative_delta"]
        ) <= 0.20,
        "best_distractor_overlap_mean_within_20pct": abs(
            metrics["best_distractor_query_overlap_mean"]["relative_delta"]
        ) <= 0.20,
        "lexical_top1_rate_within_15_points": abs(
            metrics["relevant_is_lexical_top1_rate"]["absolute_delta"]
        ) <= 0.15,
        "answer_token_coverage_within_15_points": abs(
            metrics["answer_token_coverage_mean_non_abstention"]["absolute_delta"]
        ) <= 0.15,
        "sentence_prefix_unique_rate_within_15pct": abs(
            metrics["evidence_sentence_prefix_unique_rate_mean"]["relative_delta"]
        ) <= 0.15,
        "evidence_type_token_ratio_within_15pct": abs(
            metrics["evidence_type_token_ratio_mean"]["relative_delta"]
        ) <= 0.15,
        "question_type_relevant_turn_means_within_5pct": all(
            type_turn_checks.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "question_type_relevant_turn_checks": type_turn_checks,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-data",
        type=Path,
        default=Path("benchmark/longMemEval/longmemeval_s_cleaned.json"),
    )
    parser.add_argument(
        "--official-oracle",
        type=Path,
        default=Path("benchmark/longMemEval/longmemeval_oracle.json"),
    )
    parser.add_argument(
        "--synthetic-data",
        type=Path,
        default=Path(
            "benchmark/longMemEval/synthetic_v2/"
            "pragmos_synthetic_longmemeval_500_v2.json"
        ),
    )
    parser.add_argument(
        "--synthetic-audit",
        type=Path,
        default=Path(
            "benchmark/longMemEval/synthetic_v2/"
            "pragmos_synthetic_longmemeval_500_v2_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark/longMemEval/synthetic_v2/"
            "pragmos_synthetic_longmemeval_500_v2_semantic_report.json"
        ),
    )
    args = parser.parse_args()

    official_records = load_json(args.official_data)
    oracle_records = load_json(args.official_oracle)
    synthetic_records = load_json(args.synthetic_data)
    official = dataset_semantics(
        official_records,
        official_relevance_map(official_records, oracle_records),
    )
    synthetic = dataset_semantics(
        synthetic_records,
        synthetic_relevance_map(args.synthetic_audit),
    )
    report = {
        "schema_version": 1,
        "method": (
            "Oracle answer-bearing turn labels mapped onto the complete official "
            "haystack; synthetic labels read from the private audit sidecar."
        ),
        "official": official,
        "synthetic": synthetic,
        "comparison": compare(official, synthetic),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote semantic report: {args.output}")
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
