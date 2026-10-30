#!/usr/bin/env python3
"""Prepare independent UltraChat background text for LongMemEval synthetic V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import nltk
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DATASET_SPLIT = "train_sft"
DATASET_LICENSE = "MIT"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
TOPIC_SEED = 20261026
TOPIC_ALLOWED_TAGS = {"JJ", "JJR", "JJS", "NN", "NNS"}
TOPIC_NOUN_TAGS = {"NN", "NNS"}
TOPIC_META_WORDS = {
    "add", "analysis", "analyze", "answer", "article", "assistant", "based",
    "blog", "clarify", "compare", "comparison", "comprehensive", "create",
    "describe", "description", "detail", "details", "discuss", "elaborate",
    "essay", "example", "examples", "explain", "find", "following", "format",
    "generate", "give", "given", "good", "great", "help", "include", "information",
    "instructions", "interesting", "list", "material", "mentioned", "paragraph",
    "passage", "please", "post", "previously", "provide", "question", "recommend",
    "report", "request", "response", "sample", "sentence", "specific", "steps", "story",
    "suggest", "summary", "sure", "tell", "text", "thank", "thanks", "thorough",
    "translate", "user", "various", "ways", "website", "write",
}
TOPIC_STOPWORDS = set(ENGLISH_STOP_WORDS) | TOPIC_META_WORDS
TOPIC_REJECT_TOKENS = {
    "affect", "affects", "com", "determine", "determines", "did", "does",
    "ensure", "ensures", "feels", "has", "have", "html", "http", "https",
    "incorporate", "incorporates", "incorporating", "means", "overcame",
    "perform", "performs", "reveals", "said", "sank", "says", "shouldn't",
    "transform", "transforms", "use", "uses", "utilizing", "work", "works",
    "www", "n't", "i'm", "i've", "i'll", "you're",
}


def word_count(text: object) -> int:
    return len(WORD_RE.findall(str(text or "")))


def normalized_hash(text: object) -> str:
    normalized = " ".join(str(text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _topic_candidates(content: str) -> list[tuple[str, tuple[str, ...], float]]:
    tokens = WORD_RE.findall(content)[:180]
    if len(tokens) < 2:
        return []
    tagged = nltk.pos_tag(tokens, lang="eng")
    spans: list[list[tuple[str, str, int]]] = []
    current: list[tuple[str, str, int]] = []
    for position, (token, tag) in enumerate(tagged):
        value = token.lower()
        eligible = (
            tag in TOPIC_ALLOWED_TAGS
            and value not in TOPIC_STOPWORDS
            and value not in TOPIC_REJECT_TOKENS
            and len(value) >= 3
            and not value.endswith("'s")
        )
        if eligible:
            current.append((value, tag, position))
        else:
            if len(current) >= 2:
                spans.append(current)
            current = []
    if len(current) >= 2:
        spans.append(current)

    candidates: list[tuple[str, tuple[str, ...], float]] = []
    for span in spans:
        for length in range(min(4, len(span)), 1, -1):
            for start in range(len(span) - length + 1):
                window = span[start:start + length]
                if window[-1][1] not in TOPIC_NOUN_TAGS:
                    continue
                values = tuple(value for value, _, _ in window)
                phrase = " ".join(values)
                if len(phrase) > 64 or len(set(values)) != len(values):
                    continue
                position = window[0][2]
                score = (
                    min(length, 3) * 3.0
                    + (1.0 if length == 3 else 0.0)
                    - position * 0.01
                )
                candidates.append((phrase, values, score))
    return candidates


def extract_topic_phrases(user_path: Path, limit: int = 3000) -> list[str]:
    pool: list[tuple[str, tuple[str, ...], float]] = []
    with user_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            content = str(json.loads(line).get("content") or "")
            candidates = _topic_candidates(content)
            if candidates:
                pool.append(max(candidates, key=lambda value: value[2]))
            if len(pool) >= max(15000, limit * 5):
                break

    random.Random(TOPIC_SEED).shuffle(pool)
    topics: list[str] = []
    seen_phrases: set[str] = set()
    token_use: Counter[str] = Counter()
    while pool and len(topics) < limit:
        batch, pool = pool[:80], pool[80:]
        batch.sort(
            key=lambda value: (
                sum(token_use[token] == 0 for token in value[1]),
                -sum(token_use[token] for token in value[1]),
                value[2],
            ),
            reverse=True,
        )
        for phrase, phrase_tokens, _ in batch[:16]:
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            topics.append(phrase)
            token_use.update(set(phrase_tokens))
            if len(topics) >= limit:
                break
    if len(topics) < limit:
        raise RuntimeError(
            f"Only extracted {len(topics)} independent topic phrases from "
            f"{len(seen_phrases)} unique candidates"
        )
    return topics


def official_turn_hashes_and_budget(path: Path, margin: float) -> tuple[set[str], dict[str, int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    hashes: set[str] = set()
    budgets: Counter[str] = Counter()
    for record in records:
        for session in record.get("haystack_sessions") or []:
            for message in session:
                role = str(message.get("role") or "")
                content = str(message.get("content") or "")
                hashes.add(normalized_hash(content))
                if role in {"user", "assistant"}:
                    budgets[role] += min(word_count(content), 900)
    return hashes, {
        role: int(count * margin) for role, count in budgets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-data",
        type=Path,
        default=Path("benchmark/longMemEval/longmemeval_s_cleaned.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/longMemEval/synthetic_v2/background"),
    )
    parser.add_argument(
        "--nltk-data-dir",
        type=Path,
        default=Path("benchmark/longMemEval/synthetic_v2/nltk_data"),
    )
    parser.add_argument("--margin", type=float, default=1.08)
    parser.add_argument(
        "--topics-only",
        action="store_true",
        help="Regenerate topic phrases from an existing user background pool.",
    )
    args = parser.parse_args()
    if args.margin < 1.0:
        raise ValueError("--margin must be at least 1.0")
    nltk.data.path.insert(0, str(args.nltk_data_dir.resolve()))
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK averaged_perceptron_tagger_eng data. Download it with: "
            f"python -m nltk.downloader -d {args.nltk_data_dir} "
            "averaged_perceptron_tagger_eng"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "user": args.output_dir / "ultrachat_user_background.jsonl",
        "assistant": args.output_dir / "ultrachat_assistant_background.jsonl",
    }
    topics_path = args.output_dir / "ultrachat_topic_phrases.json"
    manifest_path = args.output_dir / "ultrachat_background_manifest.json"
    if args.topics_only:
        if not paths["user"].is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "--topics-only requires an existing user pool and background manifest"
            )
        topics = extract_topic_phrases(paths["user"])
        topics_path.write_text(
            json.dumps(topics, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["topic_phrases"] = {
            "path": topics_path.name,
            "sha256": sha256_path(topics_path),
            "count": len(topics),
            "extractor": "nltk_pos_noun_phrases_v1",
            "extractor_seed": TOPIC_SEED,
            "nltk_resource": "averaged_perceptron_tagger_eng",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote topic phrases: {topics_path}")
        print(json.dumps(manifest["topic_phrases"], indent=2, sort_keys=True))
        return

    official_hashes, targets = official_turn_hashes_and_budget(
        args.official_data, args.margin
    )
    temp_paths = {role: path.with_suffix(path.suffix + ".tmp") for role, path in paths.items()}
    handles = {role: path.open("w", encoding="utf-8") for role, path in temp_paths.items()}
    collected: Counter[str] = Counter()
    message_counts: Counter[str] = Counter()
    rejected_official_overlap = 0
    seen_hashes: set[str] = set()
    rows_seen = 0
    try:
        stream = load_dataset(
            DATASET_NAME,
            revision=DATASET_REVISION,
            split=DATASET_SPLIT,
            streaming=True,
        )
        for row in stream:
            rows_seen += 1
            prompt_id = str(row.get("prompt_id") or "")
            for message_index, message in enumerate(row.get("messages") or []):
                role = str(message.get("role") or "")
                if role not in handles or collected[role] >= targets[role]:
                    continue
                content = str(message.get("content") or "").strip()
                count = word_count(content)
                if count < 5:
                    continue
                digest = normalized_hash(content)
                if digest in official_hashes:
                    rejected_official_overlap += 1
                    continue
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                handles[role].write(json.dumps({
                    "content": content,
                    "source_prompt_id": prompt_id,
                    "source_message_index": message_index,
                    "word_count": count,
                    "normalized_sha256": digest,
                }, ensure_ascii=True, separators=(",", ":")) + "\n")
                collected[role] += count
                message_counts[role] += 1
            if all(collected[role] >= targets[role] for role in handles):
                break
            if rows_seen % 10000 == 0:
                print(
                    f"[background] rows={rows_seen} "
                    f"user_words={collected['user']}/{targets['user']} "
                    f"assistant_words={collected['assistant']}/{targets['assistant']}",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    missing = {
        role: targets[role] - collected[role]
        for role in targets if collected[role] < targets[role]
    }
    if missing:
        raise RuntimeError(f"UltraChat stream ended before targets were met: {missing}")
    for role, temp_path in temp_paths.items():
        temp_path.replace(paths[role])

    topics = extract_topic_phrases(paths["user"])
    topics_path.write_text(
        json.dumps(topics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "purpose": "independent_background_text_for_pragmos_synthetic_longmemeval_v2",
        "source_dataset": DATASET_NAME,
        "source_revision": DATASET_REVISION,
        "source_split": DATASET_SPLIT,
        "source_license": DATASET_LICENSE,
        "official_data_sha256": sha256_path(args.official_data),
        "official_turn_hash_count": len(official_hashes),
        "rejected_exact_official_turn_overlap_count": rejected_official_overlap,
        "rows_streamed": rows_seen,
        "target_words": targets,
        "collected_words": dict(collected),
        "message_counts": dict(message_counts),
        "files": {
            role: {
                "path": path.name,
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
            for role, path in paths.items()
        },
        "topic_phrases": {
            "path": topics_path.name,
            "sha256": sha256_path(topics_path),
            "count": len(topics),
            "extractor": "nltk_pos_noun_phrases_v1",
            "extractor_seed": TOPIC_SEED,
            "nltk_resource": "averaged_perceptron_tagger_eng",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote background manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
