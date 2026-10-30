#!/usr/bin/env python3
"""Compare synthetic LoCoMo difficulty proxies with the pinned official set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
SESSION_RE = re.compile(r"session_(\d+)$")


def words(value: object) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(str(value or ""))]


def normalized(value: object) -> str:
    return " ".join(words(value))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[p90_index],
        "max": max(values),
    }


def conversation_sessions(sample: dict) -> list[tuple[int, list[dict]]]:
    sessions = []
    for key, value in sample.get("conversation", {}).items():
        match = SESSION_RE.fullmatch(key)
        if match and isinstance(value, list):
            sessions.append((int(match.group(1)), value))
    return sorted(sessions)


def analyze_dataset(dataset: list[dict]) -> dict:
    category_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    session_counts: list[int] = []
    turn_counts: list[int] = []
    conversation_words: list[int] = []
    turn_words: list[int] = []
    all_turn_texts: list[str] = []
    invalid_evidence_ids = 0
    fields: dict[int, dict[str, object]] = defaultdict(
        lambda: {
            "question_words": [],
            "answer_words": [],
            "evidence_count": [],
            "evidence_sessions": [],
            "evidence_turn_words": [],
            "question_trigrams": [],
            "question_prefixes": Counter(),
            "answers": set(),
            "answer_exact": [],
            "answer_coverage": [],
            "lexical_top1": [],
            "best_relevant_overlap": [],
            "distractors_ge_relevant": [],
            "caption_questions": 0,
            "caption_only_answers": 0,
        }
    )

    for sample in dataset:
        sessions = conversation_sessions(sample)
        turns = [turn for _, session in sessions for turn in session]
        dialog_map = {
            str(turn.get("dia_id")): (session_number, turn)
            for session_number, session in sessions
            for turn in session
        }
        session_counts.append(len(sessions))
        turn_counts.append(len(turns))
        conversation_words.append(
            sum(
                len(words(turn.get("text")))
                + len(words(turn.get("blip_caption")))
                for turn in turns
            )
        )
        for turn in turns:
            text = str(turn.get("text") or "")
            all_turn_texts.append(normalized(text))
            turn_words.append(len(words(text)))

        for qa in sample.get("qa") or []:
            category = int(qa["category"])
            category_counts[category] += 1
            family = str(qa.get("synthetic_family") or "")
            if family:
                family_counts[family] += 1
            bucket = fields[category]
            question = str(qa.get("question") or "")
            answer = qa.get("answer")
            question_tokens = words(question)
            answer_tokens = words(answer)
            bucket["question_words"].append(len(question_tokens))
            bucket["answer_words"].append(len(answer_tokens))
            bucket["answers"].add(" ".join(answer_tokens))
            bucket["question_trigrams"].extend(
                zip(question_tokens, question_tokens[1:], question_tokens[2:])
            )
            bucket["question_prefixes"][" ".join(question_tokens[:4])] += 1

            evidence_ids = [str(item) for item in qa.get("evidence") or []]
            bucket["evidence_count"].append(len(evidence_ids))
            evidence_rows = []
            for evidence_id in evidence_ids:
                row = dialog_map.get(evidence_id)
                if row is None:
                    invalid_evidence_ids += 1
                    continue
                evidence_rows.append(row)
            bucket["evidence_sessions"].append(
                len({session_number for session_number, _ in evidence_rows})
            )
            bucket["evidence_turn_words"].extend(
                len(words(turn.get("text")))
                + len(words(turn.get("blip_caption")))
                for _, turn in evidence_rows
            )
            if any(turn.get("blip_caption") for _, turn in evidence_rows):
                bucket["caption_questions"] += 1

            evidence_text = " ".join(
                f"{turn.get('text') or ''} {turn.get('blip_caption') or ''}"
                for _, turn in evidence_rows
            )
            plain_evidence_text = " ".join(
                str(turn.get("text") or "") for _, turn in evidence_rows
            )
            normalized_answer = " ".join(answer_tokens)
            normalized_evidence = normalized(evidence_text)
            if answer_tokens and category != 5:
                bucket["answer_exact"].append(
                    int(normalized_answer in normalized_evidence)
                )
                answer_vocabulary = set(answer_tokens)
                bucket["answer_coverage"].append(
                    len(answer_vocabulary & set(words(evidence_text)))
                    / max(1, len(answer_vocabulary))
                )
                if (
                    normalized_answer in normalized_evidence
                    and normalized_answer not in normalized(plain_evidence_text)
                ):
                    bucket["caption_only_answers"] += 1

            if evidence_rows and category != 5:
                query_tokens = set(question_tokens)
                relevant_ids = set(evidence_ids)
                scores = []
                for turn in turns:
                    turn_tokens = set(
                        words(
                            f"{turn.get('text') or ''} "
                            f"{turn.get('blip_caption') or ''}"
                        )
                    )
                    score = len(query_tokens & turn_tokens) / max(
                        1, len(query_tokens)
                    )
                    scores.append(
                        (score, str(turn.get("dia_id")) in relevant_ids)
                    )
                scores.sort(reverse=True)
                relevant_scores = [score for score, relevant in scores if relevant]
                if relevant_scores:
                    best_relevant = max(relevant_scores)
                    bucket["lexical_top1"].append(int(scores[0][1]))
                    bucket["best_relevant_overlap"].append(best_relevant)
                    bucket["distractors_ge_relevant"].append(
                        sum(
                            1
                            for score, relevant in scores
                            if not relevant and score >= best_relevant
                        )
                    )

    normalized_turns = [value for value in all_turn_texts if value]
    turn_trigrams = [
        trigram
        for value in normalized_turns
        for trigram in zip(value.split(), value.split()[1:], value.split()[2:])
    ]
    report = {
        "conversation_count": len(dataset),
        "question_count": sum(category_counts.values()),
        "category_counts": dict(sorted(category_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "session_count": describe(session_counts),
        "turn_count": describe(turn_counts),
        "conversation_words": describe(conversation_words),
        "turn_words": describe(turn_words),
        "unique_turn_rate": len(set(normalized_turns)) / max(1, len(normalized_turns)),
        "turn_trigram_diversity": len(set(turn_trigrams)) / max(1, len(turn_trigrams)),
        "invalid_evidence_id_count": invalid_evidence_ids,
        "by_category": {},
    }
    for category in sorted(category_counts):
        bucket = fields[category]
        trigrams = bucket["question_trigrams"]
        answer_count = category_counts[category]
        report["by_category"][str(category)] = {
            "question_count": answer_count,
            "question_words": describe(bucket["question_words"]),
            "answer_words": describe(bucket["answer_words"]),
            "evidence_count": describe(bucket["evidence_count"]),
            "evidence_sessions": describe(bucket["evidence_sessions"]),
            "evidence_turn_words": describe(bucket["evidence_turn_words"]),
            "question_trigram_diversity": len(set(trigrams)) / max(1, len(trigrams)),
            "unique_answer_rate": len(bucket["answers"]) / max(1, answer_count),
            "top_question_prefixes": bucket["question_prefixes"].most_common(8),
            "answer_exact_in_evidence_rate": (
                statistics.mean(bucket["answer_exact"])
                if bucket["answer_exact"]
                else None
            ),
            "answer_token_coverage_mean": (
                statistics.mean(bucket["answer_coverage"])
                if bucket["answer_coverage"]
                else None
            ),
            "lexical_relevant_top1_rate": (
                statistics.mean(bucket["lexical_top1"])
                if bucket["lexical_top1"]
                else None
            ),
            "best_relevant_query_overlap_mean": (
                statistics.mean(bucket["best_relevant_overlap"])
                if bucket["best_relevant_overlap"]
                else None
            ),
            "distractors_at_or_above_best_relevant_mean": (
                statistics.mean(bucket["distractors_ge_relevant"])
                if bucket["distractors_ge_relevant"]
                else None
            ),
            "caption_evidence_question_count": bucket["caption_questions"],
            "caption_only_answer_count": bucket["caption_only_answers"],
        }
    return report


def relative_delta(actual: float, expected: float) -> float:
    return (actual - expected) / max(abs(expected), 1e-9)


def compare_reports(official: dict, synthetic: dict) -> dict:
    checks = {
        "conversation_count_exact": synthetic["conversation_count"]
        == official["conversation_count"],
        "question_count_exact": synthetic["question_count"]
        == official["question_count"],
        "category_counts_exact": synthetic["category_counts"]
        == official["category_counts"],
        "invalid_evidence_ids_zero": synthetic["invalid_evidence_id_count"] == 0,
        "conversation_words_mean_within_15pct": abs(
            relative_delta(
                synthetic["conversation_words"]["mean"],
                official["conversation_words"]["mean"],
            )
        )
        <= 0.15,
        "turn_words_mean_within_15pct": abs(
            relative_delta(
                synthetic["turn_words"]["mean"],
                official["turn_words"]["mean"],
            )
        )
        <= 0.15,
        "unique_turn_rate_within_10_points": abs(
            synthetic["unique_turn_rate"] - official["unique_turn_rate"]
        )
        <= 0.10,
        "turn_trigram_diversity_at_least_75pct": synthetic[
            "turn_trigram_diversity"
        ]
        >= official["turn_trigram_diversity"] * 0.75,
    }
    category_checks = {}
    for category, official_values in official["by_category"].items():
        synthetic_values = synthetic["by_category"][category]
        values = {
            "question_words_mean_within_25pct": abs(
                relative_delta(
                    synthetic_values["question_words"]["mean"],
                    official_values["question_words"]["mean"],
                )
            )
            <= 0.25,
            "evidence_count_mean_within_5pct": abs(
                relative_delta(
                    synthetic_values["evidence_count"]["mean"],
                    official_values["evidence_count"]["mean"],
                )
            )
            <= 0.05,
            "evidence_sessions_mean_within_5pct": abs(
                relative_delta(
                    synthetic_values["evidence_sessions"]["mean"],
                    official_values["evidence_sessions"]["mean"],
                )
            )
            <= 0.05,
            "question_trigram_diversity_at_least_70pct": synthetic_values[
                "question_trigram_diversity"
            ]
            >= official_values["question_trigram_diversity"] * 0.70,
            "unique_answer_rate_within_20_points": (
                category == "5"
                or synthetic_values["unique_answer_rate"]
                >= official_values["unique_answer_rate"] - 0.20
            ),
            "caption_question_count_exact": synthetic_values[
                "caption_evidence_question_count"
            ]
            == official_values["caption_evidence_question_count"],
        }
        if category != "5":
            values.update(
                {
                    "answer_words_mean_within_30pct": abs(
                        relative_delta(
                            synthetic_values["answer_words"]["mean"],
                            official_values["answer_words"]["mean"],
                        )
                    )
                    <= 0.30,
                    "answer_coverage_within_20_points": abs(
                        synthetic_values["answer_token_coverage_mean"]
                        - official_values["answer_token_coverage_mean"]
                    )
                    <= 0.20,
                    "lexical_top1_within_25_points": abs(
                        synthetic_values["lexical_relevant_top1_rate"]
                        - official_values["lexical_relevant_top1_rate"]
                    )
                    <= 0.25,
                    "best_relevant_overlap_within_30pct": abs(
                        relative_delta(
                            synthetic_values[
                                "best_relevant_query_overlap_mean"
                            ],
                            official_values[
                                "best_relevant_query_overlap_mean"
                            ],
                        )
                    )
                    <= 0.30,
                }
            )
        category_checks[category] = values
    return {
        "checks": checks,
        "category_checks": category_checks,
        "passed": all(checks.values())
        and all(
            all(values.values()) for values in category_checks.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-data",
        type=Path,
        default=Path("benchmark/locomo/official/locomo10.json"),
    )
    parser.add_argument(
        "--synthetic-data",
        type=Path,
        default=Path(
            "benchmark/locomo/synthetic_v2/"
            "pragmos_synthetic_locomo_1986_v2.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark/locomo/synthetic_v2/"
            "pragmos_synthetic_locomo_1986_v2_semantic_report.json"
        ),
    )
    args = parser.parse_args()
    official = analyze_dataset(
        json.loads(args.official_data.read_text(encoding="utf-8"))
    )
    synthetic = analyze_dataset(
        json.loads(args.synthetic_data.read_text(encoding="utf-8"))
    )
    comparison = compare_reports(official, synthetic)
    report = {
        "schema_version": 1,
        "official_file": str(args.official_data),
        "official_sha256": sha256_path(args.official_data),
        "synthetic_file": str(args.synthetic_data),
        "synthetic_sha256": sha256_path(args.synthetic_data),
        "official": official,
        "synthetic": synthetic,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote semantic report: {args.output}")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
