#!/usr/bin/env python3
"""Generate a high-fidelity, distribution-matched LongMemEval development set.

V2 uses the official file only to derive anonymous structural profiles: question
type/subtype, abstention flag, timestamps, session/turn shapes, role patterns,
and question/answer length targets. Official questions, answers, entities, and
conversation text are never emitted into the generated dataset.

This is development data, not an official LongMemEval score and not an
independent evaluation set. The audit sidecar contains evidence locations and
must never be supplied to an inference system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 20261026
SCHEMA_VERSION = 2
INSUFFICIENT_ANSWER = "I do not know."
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
NUMERIC_MENTION_RE = re.compile(
    r"(?<!\w)(?:[$\u00a3\u20ac]\s*)?\d+(?:[.,:]\d+)?(?:\s*%|\s*-\s*\d+)?"
)
QUERY_ANCHOR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "that",
    "the", "their", "them", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "with", "would", "you", "your",
}
TIMESTAMP_FORMAT = "%Y/%m/%d (%a) %H:%M"

QUESTION_TYPES = {
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
}
HARD_DISTRACTOR_RATES = {
    "knowledge-update": 0.72,
    "multi-session": 0.60,
    "single-session-assistant": 0.22,
    "single-session-preference": 0.90,
    "single-session-user": 0.60,
    "temporal-reasoning": 0.45,
}

PEOPLE = [
    "Aarav", "Bianca", "Caleb", "Dalia", "Emiko", "Farah", "Gideon",
    "Hana", "Imani", "Jonas", "Keira", "Luca", "Maya", "Nikhil",
    "Odette", "Priya", "Rafael", "Sofia", "Tariq", "Yuna",
]
PLACES = [
    "Aster Hall", "Beacon Workshop", "Cedar House", "Driftwood Studio",
    "Elm Street Library", "Fern Court", "Granite Center", "Harbor Annex",
    "Indigo Room", "Juniper Pavilion", "Kestrel Arts", "Lantern Theatre",
    "Maple Commons", "Northwind Lab", "Orchid Gallery", "Pioneer Hall",
]
OBJECTS = [
    "field recorder", "ceramic lantern", "travel tripod", "mechanical keyboard",
    "watercolor set", "camping stove", "silver pendant", "folding bicycle",
    "espresso grinder", "vintage camera", "garden bench", "acoustic guitar",
]
ACTIVITIES = [
    "bookbinding", "trail running", "portrait photography", "bread baking",
    "language exchange", "community gardening", "model building", "pottery",
    "open-water swimming", "volunteer tutoring", "woodworking", "calligraphy",
]
EVENTS = [
    "the riverside workshop", "the evening lecture", "the spring exhibition",
    "the neighborhood fundraiser", "the design review", "the coastal hike",
    "the chamber concert", "the makers' fair", "the library reading",
    "the conservation meeting", "the family reunion", "the training seminar",
]
EVENT_KINDS = [
    "orientation", "planning session", "workshop", "field visit", "review",
    "demonstration", "community meeting", "training day", "presentation",
    "assessment", "open house", "follow-up", "site tour", "practice session",
    "public lecture", "team briefing", "studio visit", "final showcase",
    "research interview", "weekend seminar", "design critique", "trial run",
    "volunteer shift", "committee hearing", "project launch", "closing session",
]
ORGANIZATIONS = [
    "Northwind Collective", "Cedar Analytics", "Harbor Arts Council",
    "Juniper Health Network", "Aster Research Group", "Beacon Community Trust",
]
PRODUCTS = [
    "Aurora K2", "Boreal Mini", "Cobalt Seven", "Dune Pro", "Ember One",
    "Fjord Lite", "Grove X4", "Helio Plus",
]
COLORS = [
    "cobalt blue", "forest green", "marigold yellow", "plum purple",
    "coral red", "silver gray", "deep teal", "warm ivory",
]
TITLES = [
    "The Clockmaker's Map", "Winter at Marigold Station",
    "The Paper Observatory", "Signals Across the Lake",
    "A Garden of Small Moons", "The Last Tram Home",
    "Threads of the Monsoon", "The Quiet Cartographer",
]

TOPIC_PACKS = [
    ("urban gardening", "seedlings", "balcony", "watering schedule"),
    ("rail travel", "timetable", "platform", "connection"),
    ("home cooking", "recipe", "pantry", "preparation"),
    ("digital photography", "lens", "lighting", "composition"),
    ("community theater", "rehearsal", "stage", "performance"),
    ("software testing", "release", "workflow", "defect report"),
    ("local history", "archive", "exhibition", "oral account"),
    ("fitness planning", "routine", "recovery", "training log"),
    ("language learning", "vocabulary", "conversation", "practice session"),
    ("woodworking", "joinery", "workbench", "finishing method"),
    ("budget travel", "reservation", "neighborhood", "daily itinerary"),
    ("music practice", "arrangement", "tempo", "recording session"),
    ("book collecting", "edition", "bookshelf", "catalog entry"),
    ("volunteer work", "roster", "community center", "outreach plan"),
    ("ceramics", "glaze", "kiln", "studio session"),
    ("data analysis", "dataset", "notebook", "validation step"),
    ("bird watching", "field guide", "wetland", "observation note"),
    ("apartment repairs", "fixture", "toolbox", "maintenance visit"),
    ("public speaking", "outline", "audience", "practice recording"),
    ("museum visits", "gallery", "curator", "exhibit note"),
    ("cycling", "route", "repair stand", "distance log"),
    ("film editing", "timeline", "sound mix", "export setting"),
    ("career planning", "application", "interview", "skills inventory"),
    ("pet care", "feeding plan", "clinic", "activity record"),
]


@dataclass(frozen=True)
class RelevantEvidenceTarget:
    words: int
    sentences: int
    numeric_mentions: int


@dataclass(frozen=True)
class StructuralProfile:
    source_index: int
    question_type: str
    subtype: str
    abstention: bool
    question_words: int
    answer_words: int
    question_date: datetime
    session_dates: list[datetime]
    session_roles: list[list[str]]
    session_word_counts: list[list[int]]
    answer_session_indices: list[int]
    relevant_evidence_target: RelevantEvidenceTarget | None = None
    target_relevant_turn_count: int | None = None
    suppress_relevant_evidence: bool = False


@dataclass(frozen=True)
class EvidenceSpec:
    answer_ordinal: int
    role: str
    text: str
    evidence_type: str


@dataclass(frozen=True)
class Scenario:
    question: str
    answer: str | int | float
    family: str
    topic: str
    evidence: list[EvidenceSpec]
    near_distractors: list[str]


class BackgroundProvider:
    """Consume independent source sentences without reusing official text."""

    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, user_path: Path, assistant_path: Path) -> None:
        self.paths = {"user": user_path, "assistant": assistant_path}
        self.handles = {
            role: path.open("r", encoding="utf-8") for role, path in self.paths.items()
        }
        self.fragments: dict[str, deque[str]] = {
            "user": deque(), "assistant": deque()
        }
        self.source_messages = Counter()
        self.topic_fragments: dict[tuple[str, str], list[str]] = {}

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def _load_fragments(self, role: str) -> None:
        line = self.handles[role].readline()
        if not line:
            raise RuntimeError(f"Background pool exhausted for role={role}")
        content = str(json.loads(line).get("content") or "").strip()
        fragments = [
            value.strip() for value in self.SENTENCE_RE.split(content) if value.strip()
        ]
        self.fragments[role].extend(fragments or [content])
        self.source_messages[role] += 1

    @staticmethod
    def _split_words(text: str, count: int) -> tuple[str, str]:
        matches = list(WORD_RE.finditer(text))
        if len(matches) <= count:
            return text, ""
        end = matches[count - 1].end()
        return text[:end].rstrip(), text[end:].lstrip()

    def _take_plain(self, role: str, target_words: int) -> str:
        if target_words <= 0:
            return ""
        pieces: list[str] = []
        current = 0
        while current < target_words:
            if not self.fragments[role]:
                self._load_fragments(role)
            fragment = self.fragments[role].popleft()
            remaining = target_words - current
            count = word_count(fragment)
            if count <= remaining:
                pieces.append(fragment)
                current += count
            else:
                head, tail = self._split_words(fragment, remaining)
                pieces.append(head)
                if tail:
                    self.fragments[role].appendleft(tail)
                current = target_words
        return " ".join(pieces).strip()

    def take(self, role: str, target_words: int, required_text: str | None = None) -> str:
        if role not in self.handles:
            role = "user"
        if not required_text:
            return self._take_plain(role, target_words)
        required = required_text.strip()
        required_words = word_count(required)
        target = max(target_words, required_words + 12)
        available = target - required_words
        prefix_words = min(max(5, target // 3), max(5, available // 2))
        suffix_words = max(0, available - prefix_words)
        return " ".join(part for part in [
            self._take_plain(role, prefix_words),
            required,
            self._take_plain(role, suffix_words),
        ] if part)

    def take_sentence_fragments(
        self,
        role: str,
        word_targets: list[int],
    ) -> list[str]:
        """Select intact source sentences close to requested lengths."""
        if role not in self.handles:
            role = "user"
        selected = []
        for target in word_targets:
            candidates = []
            while len(candidates) < 24:
                if not self.fragments[role]:
                    self._load_fragments(role)
                candidates.append(self.fragments[role].popleft())
            best_index = min(
                range(len(candidates)),
                key=lambda index: (
                    abs(word_count(candidates[index]) - target),
                    word_count(candidates[index]) < target,
                    index,
                ),
            )
            selected.append(candidates.pop(best_index))
            self.fragments[role].extendleft(reversed(candidates))
        return selected

    def index_topic_contexts(self, topic_phrases: list[str]) -> None:
        """Index independent messages sharing each selected topic's conversation."""
        normalized_topics = {
            topic: " ".join(value.lower() for value in words(topic))
            for topic in topic_phrases
        }
        topics_by_first_token: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for topic, normalized in normalized_topics.items():
            if normalized:
                topics_by_first_token[normalized.split()[0]].append(
                    (topic, normalized)
                )

        topic_prompt_ids: dict[str, str] = {}
        with self.paths["user"].open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                content_tokens = [
                    value.lower() for value in words(item.get("content"))
                ]
                token_set = set(content_tokens)
                normalized_content = " ".join(content_tokens)
                for first_token in token_set & topics_by_first_token.keys():
                    for topic, normalized in topics_by_first_token[first_token]:
                        if topic in topic_prompt_ids:
                            continue
                        if normalized in normalized_content:
                            topic_prompt_ids[topic] = str(item["source_prompt_id"])
                if len(topic_prompt_ids) == len(normalized_topics):
                    break

        topics_by_prompt: dict[str, list[str]] = defaultdict(list)
        for topic, prompt_id in topic_prompt_ids.items():
            topics_by_prompt[prompt_id].append(topic)
        collected: dict[tuple[str, str], list[str]] = defaultdict(list)
        for role, path in self.paths.items():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    prompt_id = str(item.get("source_prompt_id") or "")
                    topics = topics_by_prompt.get(prompt_id)
                    if not topics:
                        continue
                    fragments = [
                        value.strip()
                        for value in self.SENTENCE_RE.split(
                            str(item.get("content") or "").strip()
                        )
                        if value.strip()
                    ]
                    for topic in topics:
                        collected[(topic, role)].extend(fragments)
        self.topic_fragments = dict(collected)

    def topic_context_summary(self, topic_phrases: list[str]) -> dict:
        topics = list(dict.fromkeys(topic_phrases))
        user_topics = {
            topic for topic in topics if self.topic_fragments.get((topic, "user"))
        }
        assistant_topics = {
            topic
            for topic in topics
            if self.topic_fragments.get((topic, "assistant"))
        }
        return {
            "requested_topic_count": len(topics),
            "topics_with_any_context": len(user_topics | assistant_topics),
            "topics_with_user_context": len(user_topics),
            "topics_with_assistant_context": len(assistant_topics),
            "topics_with_both_roles": len(user_topics & assistant_topics),
            "indexed_user_sentence_count": sum(
                len(self.topic_fragments.get((topic, "user"), []))
                for topic in topics
            ),
            "indexed_assistant_sentence_count": sum(
                len(self.topic_fragments.get((topic, "assistant"), []))
                for topic in topics
            ),
        }

    @staticmethod
    def _pop_closest_fragment(
        candidates: list[str],
        target: int,
        topic_tokens: set[str],
    ) -> str | None:
        if not candidates:
            return None
        best_index = min(
            range(len(candidates)),
            key=lambda index: (
                not bool(
                    {value.lower() for value in words(candidates[index])}
                    & topic_tokens
                ),
                abs(word_count(candidates[index]) - target),
                word_count(candidates[index]) < target,
                index,
            ),
        )
        return candidates.pop(best_index)

    def take_topic_sentence_fragments(
        self,
        topic: str,
        role: str,
        word_targets: list[int],
    ) -> list[str]:
        if role not in self.handles:
            role = "user"
        opposite_role = "assistant" if role == "user" else "user"
        primary = self.topic_fragments.setdefault((topic, role), [])
        secondary = self.topic_fragments.setdefault((topic, opposite_role), [])
        topic_tokens = {
            value.lower()
            for value in words(topic)
            if len(value) >= 3 and not any(character.isdigit() for character in value)
        }
        selected = []
        for target in word_targets:
            fragment = self._pop_closest_fragment(primary, target, topic_tokens)
            if fragment is None:
                fragment = self._pop_closest_fragment(
                    secondary,
                    target,
                    topic_tokens,
                )
            if fragment is None:
                fragment = self.take_sentence_fragments(role, [target])[0]
            selected.append(fragment)
        return selected


def words(text: object) -> list[str]:
    return WORD_RE.findall(str(text or ""))


def word_count(text: object) -> int:
    return len(words(text))


def sentence_count(text: object) -> int:
    return len(
        [
            sentence
            for sentence in SENTENCE_RE.findall(str(text or ""))
            if sentence.strip()
        ]
    )


def numeric_mention_count(text: object) -> int:
    return len(NUMERIC_MENTION_RE.findall(str(text or "")))


def evidence_text_metrics(texts: Iterable[object]) -> RelevantEvidenceTarget:
    combined = " ".join(str(text or "") for text in texts)
    return RelevantEvidenceTarget(
        words=word_count(combined),
        sentences=sentence_count(combined),
        numeric_mentions=numeric_mention_count(combined),
    )


def distribute_integer_total(total: int, count: int) -> list[int]:
    """Distribute an aggregate target deterministically without changing its sum."""
    if count <= 0:
        return []
    total = max(0, int(total))
    return [
        ((index + 1) * total) // count - (index * total) // count
        for index in range(count)
    ]


def official_relevant_evidence_summary(records: list[dict]) -> dict:
    """Summarize complete official turns marked as containing answer evidence."""
    rows: dict[str, list[RelevantEvidenceTarget]] = defaultdict(list)
    turn_counts: dict[str, list[int]] = defaultdict(list)
    for record in records:
        question_type = str(record.get("question_type") or "unknown")
        evidence_turns = [
            message.get("content") or ""
            for session in record.get("haystack_sessions") or []
            for message in session
            if message.get("has_answer") is True
        ]
        rows[question_type].append(evidence_text_metrics(evidence_turns))
        turn_counts[question_type].append(len(evidence_turns))

    def stats(values: list[int]) -> dict:
        ordered = sorted(values)
        if not ordered:
            return {"mean": 0.0, "median": 0.0, "p90": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(ordered),
            "median": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "min": ordered[0],
            "max": ordered[-1],
        }

    def summarize(question_types: list[str]) -> dict:
        values = [value for question_type in question_types for value in rows[question_type]]
        turns = [value for question_type in question_types for value in turn_counts[question_type]]
        return {
            "records": len(values),
            "words": stats([value.words for value in values]),
            "sentences": stats([value.sentences for value in values]),
            "numeric_mentions": stats([value.numeric_mentions for value in values]),
            "evidence_turn_count": stats(turns),
        }

    question_types = sorted(rows)
    return {
        "overall": summarize(question_types),
        "by_question_type": {
            question_type: summarize([question_type])
            for question_type in question_types
        },
    }


def summarize_relevant_metric_rows(
    rows: list[tuple[str, RelevantEvidenceTarget, int]],
) -> dict:
    grouped: dict[str, list[tuple[RelevantEvidenceTarget, int]]] = defaultdict(list)
    for question_type, metrics, turn_count in rows:
        grouped[question_type].append((metrics, turn_count))

    def stats(values: list[int]) -> dict:
        ordered = sorted(values)
        if not ordered:
            return {"mean": 0.0, "median": 0.0, "p90": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(ordered),
            "median": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "min": ordered[0],
            "max": ordered[-1],
        }

    def summarize(question_types: list[str]) -> dict:
        values = [value for question_type in question_types for value in grouped[question_type]]
        return {
            "records": len(values),
            "words": stats([value.words for value, _turns in values]),
            "sentences": stats([value.sentences for value, _turns in values]),
            "numeric_mentions": stats(
                [value.numeric_mentions for value, _turns in values]
            ),
            "evidence_turn_count": stats([turns for _value, turns in values]),
        }

    question_types = sorted(grouped)
    return {
        "overall": summarize(question_types),
        "by_question_type": {
            question_type: summarize([question_type])
            for question_type in question_types
        },
    }


def synthetic_relevant_evidence_metrics(
    record: dict,
    audit: dict,
) -> tuple[RelevantEvidenceTarget, int]:
    sessions = dict(
        zip(record["haystack_session_ids"], record["haystack_sessions"])
    )
    locations = []
    seen = set()
    for evidence in audit.get("evidence") or []:
        location = (evidence.get("session_id"), evidence.get("turn_index"))
        if location in seen:
            continue
        seen.add(location)
        locations.append(location)
    texts = [
        sessions[session_id][int(turn_index)].get("content") or ""
        for session_id, turn_index in locations
    ]
    return evidence_text_metrics(texts), len(locations)


def build_relevant_evidence_targets(
    official_records: list[dict],
    official_evidence_summary: dict,
    seed: int,
    topic_phrases: list[str],
    relevant_turn_counts: dict[int, int],
    zero_evidence_indices: set[int],
) -> dict[int, RelevantEvidenceTarget]:
    """Match official aggregate evidence density without copying evidence text."""
    indices_by_type: dict[str, list[int]] = defaultdict(list)
    minimums_by_index: dict[int, RelevantEvidenceTarget] = {}
    for index, record in enumerate(official_records):
        question_type = str(record.get("question_type") or "unknown")
        indices_by_type[question_type].append(index)
        profile = build_profile(
            record,
            index,
            target_relevant_turn_count=relevant_turn_counts[index],
            suppress_relevant_evidence=index in zero_evidence_indices,
        )
        rng = random.Random(stable_record_seed(seed, index))
        scenario = build_scenario(
            profile,
            rng,
            topic_phrases[index % len(topic_phrases)],
        )
        minimums_by_index[index] = (
            RelevantEvidenceTarget(0, 0, 0)
            if profile.suppress_relevant_evidence
            else evidence_text_metrics(evidence.text for evidence in scenario.evidence)
        )

    targets: dict[int, RelevantEvidenceTarget] = {}
    for question_type, indices in indices_by_type.items():
        reference = official_evidence_summary["by_question_type"][question_type]
        active_indices = [
            index for index in indices if index not in zero_evidence_indices
        ]
        word_targets = allocate_targets_with_minimum(
            round(reference["words"]["mean"] * len(indices)),
            [minimums_by_index[index].words for index in active_indices],
        )
        sentence_targets = allocate_targets_with_minimum(
            round(reference["sentences"]["mean"] * len(indices)),
            [minimums_by_index[index].sentences for index in active_indices],
        )
        numeric_targets = allocate_targets_with_minimum(
            round(reference["numeric_mentions"]["mean"] * len(indices)),
            [minimums_by_index[index].numeric_mentions for index in active_indices],
        )
        for index in indices:
            if index in zero_evidence_indices:
                targets[index] = RelevantEvidenceTarget(0, 0, 0)
        for position, index in enumerate(active_indices):
            targets[index] = RelevantEvidenceTarget(
                words=word_targets[position],
                sentences=sentence_targets[position],
                numeric_mentions=numeric_targets[position],
            )
    return targets


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def format_timestamp(value: datetime) -> str:
    return value.strftime(TIMESTAMP_FORMAT)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_hash(text: object) -> str:
    normalized = " ".join(str(text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_record_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"v2:{seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def question_subtype(record: dict) -> str:
    question = str(record.get("question") or "").lower()
    question_type = record.get("question_type")
    if question_type == "multi-session":
        if re.search(r"\baverage\b", question):
            return "average"
        if re.search(r"\b(save|difference|more than|less than|compared|instead of)\b", question):
            return "difference"
        if re.search(r"\b(most|least|highest|lowest|longest|shortest|largest|smallest)\b", question):
            return "arg-extreme"
        if re.search(r"how many (different|distinct|unique)", question):
            return "count-distinct"
        if re.search(r"\b(total|altogether|combined|in all|sum)\b", question) or re.search(
            r"how much (did|have|was|were)", question
        ):
            return "sum"
        if "how many" in question:
            return "count"
        if re.search(r"\b(which|what)\b", question):
            return "collect-or-join"
        return "multi-other"
    if question_type == "temporal-reasoning":
        if re.search(r"how (many|long)", question):
            return "elapsed-duration"
        if re.search(r"\b(first|earlier|later|before|after|order)\b", question):
            return "event-order"
        if re.search(r"\bwhen\b|what date|which date|what day|what month|what year", question):
            return "date-resolution"
        return "relative-or-recent"
    if question_type == "knowledge-update":
        if re.search(r"\b(current|currently|now|latest|recent)\b", question):
            return "current-state"
        if re.search(r"\b(original|initial|first|earlier|previous|before|used to)\b", question):
            return "historical-state"
        if re.search(r"how many|how much", question):
            return "updated-quantity"
        return "updated-attribute"
    if question_type == "single-session-assistant":
        if re.search(r"\b\d+(st|nd|rd|th)\b|item \d+|number \d+", question):
            return "ordinal-list"
        if re.search(r"phone|email|url|website|contact", question):
            return "contact-detail"
        if re.search(r"recommend|suggest", question):
            return "assistant-recommendation"
        if re.search(r"list|which|what (were|are) the", question):
            return "assistant-list-or-fact"
        return "assistant-recall"
    if question_type == "single-session-preference":
        return "preference-recommendation"
    if question_type == "single-session-user":
        if question.startswith("where") or " where " in question:
            return "place"
        if question.startswith("who") or " who " in question:
            return "person"
        if re.search(r"when|what date|which date|what day|what year|how old", question):
            return "date"
        if "how long" in question:
            return "duration"
        if re.search(r"how many|how much", question):
            return "quantity"
        return "attribute"
    raise ValueError(f"Unsupported question type: {question_type!r}")


def build_profile(
    record: dict,
    index: int,
    relevant_evidence_target: RelevantEvidenceTarget | None = None,
    target_relevant_turn_count: int | None = None,
    suppress_relevant_evidence: bool = False,
) -> StructuralProfile:
    question_type = record.get("question_type")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unsupported question type at index {index}: {question_type!r}")
    session_ids = list(record.get("haystack_session_ids") or [])
    sessions = list(record.get("haystack_sessions") or [])
    dates = list(record.get("haystack_dates") or [])
    if not (len(session_ids) == len(sessions) == len(dates)):
        raise ValueError(f"Parallel session arrays differ at index {index}")
    answer_ids = set(record.get("answer_session_ids") or [])
    return StructuralProfile(
        source_index=index,
        question_type=question_type,
        subtype=question_subtype(record),
        abstention="_abs" in str(record.get("question_id") or ""),
        question_words=word_count(record.get("question")),
        answer_words=word_count(record.get("answer")),
        question_date=parse_timestamp(record["question_date"]),
        session_dates=[parse_timestamp(value) for value in dates],
        session_roles=[[str(message.get("role") or "user") for message in session] for session in sessions],
        session_word_counts=[
            [min(word_count(message.get("content")), 900) for message in session]
            for session in sessions
        ],
        answer_session_indices=[i for i, session_id in enumerate(session_ids) if session_id in answer_ids],
        relevant_evidence_target=relevant_evidence_target,
        target_relevant_turn_count=target_relevant_turn_count,
        suppress_relevant_evidence=suppress_relevant_evidence,
    )


def choose_other(rng: random.Random, values: list[str], excluded: Iterable[str]) -> str:
    blocked = set(excluded)
    candidates = [value for value in values if value not in blocked]
    return rng.choice(candidates)


def fit_question(text: str, target_words: int, rng: random.Random) -> str:
    short_prefixes = [
        "On review", "For reference", "Looking back", "From memory",
        "As recorded", "In retrospect", "For clarity", "Checking again",
        "Revisiting this", "Returning briefly", "From before", "As noted",
        "Consulting my notes", "Reviewing the history", "For accurate recall",
        "Reading back", "To confirm", "For this reminder", "On reflection",
        "Cross-checking now", "From that record", "As a reminder",
        "Reviewing this", "To be precise", "Checking the timeline",
        "Going back", "From those notes", "For the record", "To recap",
        "When verified", "After review", "For exact recall",
        "Consulting the history", "Returning to this", "As documented",
        "To reconstruct this", "Reviewing what happened", "From our history",
        "When cross-checked", "Looking at the record",
    ]
    rng.shuffle(short_prefixes)
    base_words = word_count(text)
    if base_words >= max(7, target_words - 3):
        for prefix in short_prefixes:
            if base_words + word_count(prefix) <= target_words + 2:
                stem = text.rstrip("?")
                return f"{prefix}, {stem[0].lower() + stem[1:]}?"
    if word_count(text) >= max(7, target_words - 3):
        return text
    prefixes = [
        "Looking back at the record",
        "To refresh my memory",
        "From the relevant earlier exchange",
        "According to our conversation history",
        "While reviewing those notes",
        "Going by what was recorded",
        "From the confirmed entry",
        "As a reminder from before",
        "When I revisit that discussion",
        "Based on the dated record",
        "Using the relevant conversation",
        "From the details already shared",
        "As I reconstruct the earlier account",
        "When checking the prior notes",
        "According to the supporting exchange",
        "From the applicable part of our history",
        "While cross-checking the old record",
        "Going back to that conversation",
        "Using the entry that answers this",
        "From the earlier confirmed account",
    ]
    additions = [
        "from the notes I shared",
        "according to the recorded details",
        "as stated in the prior exchange",
        "using the relevant confirmed entry",
        "in the conversation history",
        "from that earlier discussion",
        "based on the supporting record",
        "as it appeared in my notes",
        "using only the applicable evidence",
        "after checking the dated entries",
        "while preserving the exact value",
        "without substituting a related example",
        "after excluding tentative plans",
        "without borrowing another person's detail",
        "from the completed rather than planned activity",
        "as confirmed in the original account",
        "when the relevant sessions are considered",
        "after separating updates from older values",
        "using the entry tied to my own activity",
        "without treating an estimate as final",
        "after resolving repeated mentions",
        "from the portion that directly supports it",
        "as documented before the question date",
        "after distinguishing advice from my actions",
        "using the exact wording where it matters",
        "from the evidence linked to that event",
        "after keeping different people separate",
        "as supported by the earlier account",
        "without counting a future intention",
        "after comparing the relevant timestamps",
        "from the exchange where it was confirmed",
        "using the final recorded state",
        "after excluding unrelated nearby details",
        "from the history associated with this topic",
        "as recorded across the applicable discussions",
        "without confusing a draft with the final choice",
        "after retaining the original units",
        "using the evidence attached to that update",
        "from the relevant part of the timeline",
        "without merging separate events",
    ]
    rng.shuffle(prefixes)
    rng.shuffle(additions)
    stem = text.rstrip("?")
    for prefix in prefixes:
        candidate = f"{prefix}, {stem[0].lower() + stem[1:]}"
        if word_count(candidate) <= target_words + 2:
            stem = candidate
            break
    for addition in additions:
        if word_count(stem) >= target_words - 3:
            break
        candidate = f"{stem}, {addition}"
        if word_count(candidate) <= target_words + 2:
            stem = candidate
    return stem + "?"


def expanded_answer(answer: str, target_words: int, context: str) -> str:
    if answer == INSUFFICIENT_ANSWER and target_words > word_count(answer) + 3:
        result = (
            f"I do not know. The available conversations mention related details about {context}, "
            "but they do not provide the specific fact, completed event, or required value needed "
            "to answer this question."
        )
        matches = list(WORD_RE.finditer(result))
        if len(matches) <= target_words:
            return result
        end = matches[target_words - 1].end()
        clipped = result[:end].rstrip(" ,;:")
        return clipped + ("." if not clipped.endswith((".", "?", "!")) else "")
    if target_words <= word_count(answer) + 3:
        return answer
    result = (
        f"The supported answer is {answer}. This is based on confirmed details concerning {context}; "
        "earlier values, tentative plans, and facts about other people are excluded. "
        "The exact recorded names, quantities, dates, and units are preserved."
    )
    matches = list(WORD_RE.finditer(result))
    if len(matches) <= target_words:
        return result
    end = matches[target_words - 1].end()
    clipped = result[:end].rstrip(" ,;:")
    return clipped + ("." if not clipped.endswith((".", "?", "!")) else "")


def desired_evidence_turn_count(profile: StructuralProfile) -> int:
    if profile.suppress_relevant_evidence:
        return 0
    if profile.target_relevant_turn_count is not None:
        return max(1, profile.target_relevant_turn_count)
    return max(1, len(profile.answer_session_indices))


def evidence_for_profile(
    profile: StructuralProfile,
    statements: list[str],
    role: str,
    evidence_type: str,
) -> list[EvidenceSpec]:
    if not statements:
        raise ValueError("At least one evidence statement is required")
    answer_session_count = max(1, len(profile.answer_session_indices))
    return [
        EvidenceSpec(
            ordinal % answer_session_count,
            role,
            statements[min(ordinal, len(statements) - 1)],
            evidence_type,
        )
        for ordinal in range(desired_evidence_turn_count(profile))
    ]


def direct_user_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    subtype = profile.subtype
    activity = topic_phrase
    place = rng.choice(PLACES)
    person = rng.choice(PEOPLE)
    item = rng.choice(OBJECTS)
    if subtype == "place":
        answer = place
        question = rng.choice([
            f"Where did I attend the weekly {activity} sessions?",
            f"At which venue was I taking my {activity} class?",
            f"Can you recall the place where I practiced {activity}?",
            f"What location hosted my recurring {activity} sessions?",
            f"Which venue did I travel to for the {activity} program?",
            f"Where was the {activity} class I said I had enrolled in?",
            f"At what place did my Thursday {activity} practice occur?",
            f"Which named location was associated with my {activity} attendance?",
        ])
        statement = f"The class was getting difficult to schedule, but I eventually enrolled for {activity} at {place} and have been going there on Thursday evenings."
        near = [f"I bought supplies for {activity} near {choose_other(rng, PLACES, [place])}, although the class itself was elsewhere."]
    elif subtype == "person":
        answer = person
        question = rng.choice([
            f"Who helped me choose the {item}?",
            f"Which person came with me when I selected my {item}?",
            f"Who did I say advised me about the {item}?",
            f"Whose help did I rely on while making the final {item} choice?",
            f"Who accompanied me during the comparison that led to buying the {item}?",
            f"Which person participated in my selection of the {item}?",
            f"Who was present and helped when I settled on the {item}?",
            f"What person did I credit with helping choose my {item}?",
        ])
        statement = f"I spent most of Saturday comparing options for the {item}; {person} came along and helped me make the final choice."
        near = [f"{choose_other(rng, PEOPLE, [person])} later asked whether the {item} was working well."]
    elif subtype == "duration":
        answer = rng.choice(["35 minutes", "45 minutes", "one hour", "90 minutes"])
        question = rng.choice([
            f"How long does my usual journey to {place} take?",
            f"What duration did I give for the trip to {place}?",
            f"How much time do I normally spend getting to {place}?",
            f"What is the ordinary door-to-door travel time to {place}?",
            f"How many minutes does my normal route to {place} require?",
            f"What typical journey length did I record for reaching {place}?",
            f"How long is the regular trip I make to {place}?",
            f"What amount of travel time is usual when I go to {place}?",
        ])
        statement = f"After trying two routes, I found that the ordinary door-to-door journey to {place} takes {answer} each way."
        near = [f"On one disrupted morning the same trip took two hours, but that was not the usual duration."]
    elif subtype == "date":
        answer = rng.choice(["March 14", "April 22", "May 6", "June 18"])
        question = rng.choice([
            f"When did I receive the {item}?",
            f"On what date did the {activity} event take place?",
            f"What date did I mention for getting the {item}?",
            f"Which calendar date was recorded for my receipt of the {item}?",
            f"What was the confirmed day on which the {activity} event occurred?",
            f"When, by date, did the {item} actually arrive?",
            f"What date belongs to the completed {activity} event rather than its planning?",
            f"On which day did I say I had obtained the {item}?",
        ])
        statement = f"I checked the receipt again and confirmed that I received the {item} on {answer}, not during the following week as I first thought."
        near = [f"I ordered accessories for the {item} on July 2, which was a separate purchase."]
    elif subtype == "quantity":
        answer = str(rng.randint(3, 14))
        question = rng.choice([
            f"How many {activity} workshops did I complete?",
            f"What number of sessions did I attend for {activity}?",
            f"How many times did I say I went to the {activity} program?",
            f"What was my final attendance count for the {activity} workshops?",
            f"How many completed sessions are recorded for my {activity} program?",
            f"What tally did the attendance record give for {activity}?",
            f"How many separate {activity} meetings did I finish?",
            f"What total number of {activity} workshops did I actually attend?",
        ])
        statement = f"The final attendance email says I completed {answer} separate {activity} workshops over the course of the program."
        near = [f"I originally planned to attend sixteen workshops before my schedule changed."]
    else:
        answer = rng.choice(TITLES + PRODUCTS + COLORS)
        question = rng.choice([
            f"What was the name of the {activity} project I completed?",
            f"Which title did I give my latest {activity} project?",
            f"What specific name did I mention for the project connected with {activity}?",
            f"How did I name the finished project about {activity}?",
            f"Under what title did I record my completed {activity} work?",
            f"What name appears on the final version of my {activity} project?",
            f"Which project title did I settle on after finishing the {activity} work?",
            f"What did I ultimately call the project associated with {activity}?",
            f"Which name identifies the {activity} project that I said was complete?",
            f"Can you recover the final title of my project concerning {activity}?",
        ])
        statement = f"After several revisions, I finished the {activity} project and gave it the final title {answer}."
        near = [f"An earlier draft had the working title {choose_other(rng, TITLES + PRODUCTS + COLORS, [answer])}."]
    if profile.abstention:
        missing_item = choose_other(rng, OBJECTS, [item])
        question = f"What brand was the {missing_item} that I purchased?"
        answer = INSUFFICIENT_ANSWER
        statement = f"I discussed where to store my {item} and how often I use it, but I never recorded a brand for the {missing_item}."
    question = fit_question(question, profile.question_words, rng)
    answer_text = expanded_answer(str(answer), profile.answer_words, activity)
    statements = [statement] + [f"I later confirmed the same detail while reviewing my {activity} notes."]
    return Scenario(
        question, answer_text, f"user-{subtype}", activity,
        evidence_for_profile(profile, statements, "user", "direct-user-fact"),
        near,
    )


def assistant_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    subtype = profile.subtype
    topic = topic_phrase
    if subtype == "ordinal-list":
        entries = [
            "archive indexer", "caption editor", "accessibility reviewer",
            "community newsletter writer", "language tutor", "transcript auditor",
            "pattern designer", "genealogy researcher", "workshop host",
            "catalog verifier", "remote coordinator", "documentation specialist",
        ]
        position = rng.randint(5, 10)
        answer = entries[position - 1]
        question = rng.choice([
            f"In the remote-work list you made for my {topic} plans, what was item {position}?",
            f"Can you remind me of the {position}th suggestion in your numbered {topic} job list?",
            f"Which role appeared at number {position} in the list you previously provided?",
            f"What entry occupied position {position} in your earlier {topic} list?",
            f"Reading your numbered {topic} suggestions, which item was {position}th?",
            f"At number {position}, what role did your prior list contain?",
        ])
        statement = f"For your {topic} request, here is the numbered list you requested:\n" + "\n".join(
            f"{i}. {entry}" for i, entry in enumerate(entries, 1)
        )
        near = ["A separate shortlist contained editor, tutor, and coordinator in a different order."]
    elif subtype == "contact-detail":
        kind = rng.choice(["phone", "website", "email"])
        if kind == "phone":
            answer = f"+1-415-555-{rng.randint(1000, 9999)}"
        elif kind == "website":
            answer = f"https://www.{rng.choice(['cedarlearn', 'asterguide', 'harborstudy'])}.org/resources"
        else:
            answer = f"info@{rng.choice(['northwindlab', 'juniperarts', 'beacontrust'])}.org"
        question = rng.choice([
            f"What was the {kind} address you gave me for the {topic} resource?",
            f"Can you repeat the {kind} contact you previously recommended for {topic}?",
            f"Which {kind} detail did you provide when we discussed {topic}?",
            f"What exact {kind} contact appeared in your earlier {topic} response?",
            f"Which verified {kind} value did you supply for the {topic} resource?",
            f"Can you retrieve the {kind} detail from your recommendation about {topic}?",
        ])
        statement = f"For the {topic} resource we discussed, the verified {kind} contact is {answer}. Use that exact detail rather than the similarly named directory entry."
        near = [f"The general directory used a different {kind} entry that was no longer current."]
    elif subtype == "assistant-recommendation":
        answer = rng.choice(PLACES + TITLES + PRODUCTS)
        question = rng.choice([
            f"What was the specific {topic} option you recommended to me?",
            f"Which named recommendation did you make for my {topic} plans?",
            f"Can you remind me of the {topic} recommendation you gave earlier?",
            f"What primary option did your earlier guidance select for {topic}?",
            f"Which {topic} choice did you favor over the alternatives?",
            f"What named option was the main recommendation in your {topic} response?",
            f"Which selection did your advice identify as best for my {topic} constraints?",
            f"What did you specifically recommend when I asked about {topic}?",
        ])
        statement = f"Considering the {topic} constraints you described, my specific recommendation is {answer}; it fits the schedule and the practical requirements better than the alternatives."
        near = [f"I also described {choose_other(rng, PLACES + TITLES + PRODUCTS, [answer])} as an alternative, not the primary recommendation."]
    elif subtype == "assistant-list-or-fact":
        objectives = [
            "measure access barriers", "compare seasonal participation",
            "document participant feedback",
        ]
        answer = "; ".join(objectives)
        question = rng.choice([
            f"What were the three objectives you outlined for my {topic} study?",
            f"Which three goals did you give for the {topic} project?",
            f"Can you recall the complete objective list from our {topic} discussion?",
            f"What full set of objectives appeared in your {topic} project outline?",
            f"Which goals formed the three-part plan you proposed for {topic}?",
            f"Can you reproduce all the study objectives from your earlier {topic} answer?",
            f"What were the distinct aims you listed for the {topic} work?",
            f"Which complete goal list did your guidance provide for {topic}?",
        ])
        statement = f"The three {topic} project objectives are: " + "; ".join(objectives) + "."
        near = ["The later implementation checklist contained five tasks, but those were not project objectives."]
    else:
        answer = rng.choice(COLORS + ["Chapter 6", "the evening rotation", "vacuum distillation"])
        question = rng.choice([
            f"What exact detail did you give me when we discussed {topic}?",
            f"Can you remind me of the specific answer you provided about {topic}?",
            f"Looking back at your explanation of {topic}, what detail did you state?",
            f"Which precise detail appeared in your earlier explanation of {topic}?",
            f"What did your response identify as the answer concerning {topic}?",
            f"From your prior reply about {topic}, what exact value should I recall?",
            f"Can you retrieve the particular fact you stated about {topic}?",
            f"What was the definitive detail in the guidance you gave on {topic}?",
            f"Which value did your earlier {topic} explanation distinguish from its examples?",
            f"In your response concerning {topic}, what was the requested fact?",
        ])
        statement = f"In the {topic} explanation, the exact detail was {answer}. I distinguished it from the examples that followed."
        near = [f"A worked example used {choose_other(rng, COLORS + ['Chapter 4', 'the morning rotation'], [answer])}, but that was not the requested detail."]
    if profile.abstention:
        question = f"What mailing address did you give me for the {topic} organization?"
        answer = INSUFFICIENT_ANSWER
        statement = f"I provided general information about the {topic} organization but did not give a mailing address."
    question = fit_question(question, profile.question_words, rng)
    answer_text = expanded_answer(str(answer), profile.answer_words, topic)
    statements = [statement] + [f"In a follow-up I referred back to that same {topic} detail without changing it."]
    return Scenario(
        question, answer_text, f"assistant-{subtype}", topic,
        evidence_for_profile(profile, statements, "assistant", "assistant-memory"),
        near,
    )


def preference_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    _, positive, avoid, recommendation = rng.choice([
        ("travel photography", "a compact weather-sealed setup with strong low-light performance", "heavy studio-only equipment", "a lightweight weather-sealed prime lens, two spare batteries, and a compact rain cover"),
        ("a weekend hotel", "a quiet independent property near public transport with a rooftop area", "large party-oriented resorts", "a small transit-adjacent boutique hotel with quiet rooms and a usable rooftop terrace"),
        ("a fitness plan", "low-impact swimming and mobility sessions under forty minutes", "high-impact running and hour-long classes", "three short pool sessions plus two guided mobility routines each week"),
        ("a dinner reservation", "vegetarian Korean food with clearly documented nut-free preparation", "steakhouses and unclear allergy policies", "a vegetarian Korean restaurant that confirms a separate nut-free preparation area"),
        ("holiday reading", "a character-driven historical mystery below four hundred pages", "graphic horror and very long military histories", "a concise historical mystery centered on characters rather than combat or horror"),
        ("a meditation tool", "offline timers, short unguided sessions, and no social feed", "subscription-only group challenges", "an offline meditation timer with downloadable bells, flexible intervals, and no social features"),
    ])
    topic = topic_phrase
    place = rng.choice(PLACES)
    companion = rng.choice(PEOPLE)
    question = rng.choice([
        f"Given what I told you before, what would you recommend for {topic} with {companion}?",
        f"Can you make a personalized suggestion for {topic} that fits my earlier preferences?",
        f"What option for {topic} would suit the constraints I previously described?",
        f"Taking my stated likes and exclusions together, what {topic} choice fits best?",
        f"Which {topic} recommendation follows from the preferences I shared earlier?",
        f"What would be a suitable {topic} option without violating my prior constraints?",
        f"Based on both what I favor and what I avoid, what should I choose for {topic}?",
        f"Can you tailor a {topic} recommendation to the requirements in my earlier notes?",
        f"Which practical {topic} suggestion is most consistent with my recorded preferences?",
        f"What personalized choice for {topic} accounts for all the constraints I mentioned?",
    ])
    statements = [
        f"For my {topic} plans around {place} with {companion}, I strongly prefer {positive}.",
        f"For that {topic} decision, please rule out {avoid}; those options have not worked for me and should not appear in future recommendations.",
    ]
    answer = (
        f"I would recommend {recommendation}. It directly fits the stated preference for {positive} "
        f"while avoiding {avoid}, so it is more suitable than a generic alternative."
    )
    if profile.abstention:
        question = f"Which exact brand did I previously choose for {topic}?"
        answer = INSUFFICIENT_ANSWER
        statements = [
            f"For {topic}, I described preferring {positive} and avoiding {avoid}, but I never selected or named a brand."
        ]
    question = fit_question(question, profile.question_words, rng)
    preference_statements = statements
    if desired_evidence_turn_count(profile) == 1 and len(statements) > 1:
        preference_statements = [" ".join(statements)]
    evidence = evidence_for_profile(
        profile, preference_statements, "user", "preference-constraint"
    )
    answer = expanded_answer(answer, profile.answer_words, topic)
    return Scenario(
        question, answer, "preference-personalized-recommendation", topic, evidence,
        [f"A friend preferred {avoid}, but I explicitly said that choice did not apply to me."],
    )


def knowledge_update_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    subtype = profile.subtype
    attribute, values = rng.choice([
        ("usual morning drink", ["filter coffee", "masala chai", "mint tea"]),
        ("work laptop", ["Aurora K2", "Fjord Lite", "Grove X4"]),
        ("weekly class", ["pottery", "salsa", "watercolor"]),
        ("running route", ["Canal Loop", "Garden Trail", "Harbor Path"]),
        ("job title", ["Data Analyst", "Analytics Lead", "Research Manager"]),
        ("home neighborhood", ["Cedar Quarter", "Harbor Ward", "Juniper Park"]),
    ])
    answer_count = max(1, desired_evidence_turn_count(profile))
    history = [values[min(i, len(values) - 1)] for i in range(answer_count)]
    if answer_count > len(values):
        history.extend([values[-1]] * (answer_count - len(values)))
    statements = []
    for ordinal in range(answer_count):
        value = history[ordinal]
        if ordinal == 0:
            statements.append(f"In my {topic_phrase} notes, at that stage I was still using {value} as my {attribute}, which had been my established choice for some time.")
        else:
            statements.append(f"This is an update to my {topic_phrase} notes: I changed my {attribute} to {value}, replacing the value I mentioned before.")
    current = history[-1]
    previous = history[-2] if len(history) > 1 else history[0]
    original = history[0]
    if subtype == "historical-state":
        answer = rng.choice([previous, original])
        if answer == previous:
            question = rng.choice([
                f"In my {topic_phrase} notes, what was my {attribute} immediately before I changed it to {current}?",
                f"For {topic_phrase}, which value for my {attribute} came just before the latest update?",
                f"Within the {topic_phrase} updates, before {current}, what had I most recently used as my {attribute}?",
                f"What was the penultimate {attribute} in the sequence of my {topic_phrase} changes?",
                f"Which {attribute} did the newest {topic_phrase} update replace?",
                f"Immediately prior to {current}, what value was active for my {attribute}?",
                f"Looking one revision back in the {topic_phrase} history, what was my {attribute}?",
                f"What did my {attribute} read just before the final {topic_phrase} change?",
            ])
        else:
            question = rng.choice([
                f"What was the earliest recorded value for my {attribute} in the {topic_phrase} notes?",
                f"Before any of the {topic_phrase} changes, what did I use as my {attribute}?",
                f"What was my original {attribute} at the start of the {topic_phrase} updates?",
                f"Which {attribute} value began the recorded {topic_phrase} timeline?",
                f"At the outset of the {topic_phrase} history, what was my {attribute}?",
                f"What initial {attribute} was later replaced by the subsequent updates?",
                f"Going back to the first {topic_phrase} entry, what did I list for my {attribute}?",
                f"Which value was the starting state of my {attribute} before any revision?",
            ])
    elif subtype == "updated-quantity":
        increments = [rng.randint(1, 4) for _ in range(answer_count)]
        statements = [
            f"For {topic_phrase}, I completed {value} additional {attribute} entries during this reporting period."
            for value in increments
        ]
        answer = str(sum(increments))
        question = rng.choice([
            f"How many {attribute} entries have I completed across all of the {topic_phrase} updates?",
            f"What is the updated total number of my {attribute} entries for {topic_phrase}?",
            f"Adding the separate {topic_phrase} reports together, how many {attribute} entries did I complete?",
            f"What cumulative count results from all completed {attribute} entries in the {topic_phrase} reports?",
            f"Across each {topic_phrase} update, what is the sum of the {attribute} entries I completed?",
            f"After combining every reported increment, how many {attribute} entries are complete?",
            f"What final tally do the separate {topic_phrase} completion reports produce?",
            f"How large is the aggregate of my completed {attribute} entries across those updates?",
            f"When the {topic_phrase} increments are totaled, what completed count do they give?",
        ])
    else:
        answer = current
        question = rng.choice([
            f"What is the latest value I gave for my {attribute} in the {topic_phrase} notes?",
            f"After the {topic_phrase} changes I described, what is my {attribute} now?",
            f"Which {attribute} should be treated as current for {topic_phrase} rather than superseded?",
            f"What value is presently active for my {attribute} after the {topic_phrase} revisions?",
            f"Which most recent update determines my current {attribute}?",
            f"At the end of the {topic_phrase} update sequence, what is my {attribute}?",
            f"What did I last change my {attribute} to in the {topic_phrase} record?",
            f"Which value replaced all earlier entries for my {attribute}?",
            f"What current state for my {attribute} is supported by the final update?",
        ])
    if profile.abstention:
        question = f"What is my current preferred evening meal according to these updates?"
        answer = INSUFFICIENT_ANSWER
        statements = [
            f"I updated my {attribute} to {history[min(i, len(history) - 1)]}, but I did not discuss an evening meal."
            for i in range(answer_count)
        ]
    question = fit_question(question, profile.question_words, rng)
    answer_text = expanded_answer(str(answer), profile.answer_words, attribute)
    return Scenario(
        question, answer_text, f"update-{subtype}", topic_phrase,
        evidence_for_profile(profile, statements, "user", "state-update"),
        [f"{rng.choice(PEOPLE)} still uses {choose_other(rng, values, [current])}, but that is a different person's preference."],
    )


def multi_session_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    subtype = profile.subtype
    answer_count = max(1, desired_evidence_turn_count(profile))
    topic = topic_phrase
    labels = rng.sample(
        ["notebook", "tripod", "workshop", "membership", "repair", "course", "ticket", "toolkit"],
        k=min(max(answer_count, 2), 8),
    )
    if len(labels) < answer_count:
        labels.extend(f"item {i}" for i in range(len(labels) + 1, answer_count + 1))
    values = [rng.randint(12, 95) for _ in range(answer_count)]
    statements: list[str] = []
    if subtype == "count":
        items = [f"{rng.choice(COLORS).split()[0]} {label}" for label in labels[:answer_count]]
        statements = [
            f"I completed the acquisition rather than merely planning it: I brought home the {item} for my {topic} setup."
            for item in items
        ]
        answer = str(len(set(items)))
        question = rng.choice([
            f"How many items did I actually acquire for my {topic} setup?",
            f"Across the separate conversations, how many completed {topic} purchases did I mention?",
            f"What number of {topic} items did I buy or bring home?",
            f"What is the count of confirmed acquisitions in my {topic} history?",
            f"How many separate {topic} items moved beyond planning into completed purchases?",
            f"Counting only things I obtained, what total number of {topic} items is recorded?",
            f"How many completed acquisitions belong to my own {topic} setup?",
            f"What tally do the individual conversations give for {topic} items I acquired?",
            f"After excluding intended purchases, how many {topic} items did I actually get?",
            f"Across all sessions, what count of completed {topic} acquisitions is supported?",
        ])
    elif subtype == "count-distinct":
        items = labels[:max(1, answer_count - 1)] + ([labels[0]] if answer_count > 1 else [])
        items = items[:answer_count]
        statements = [f"I used the {item} in a completed {topic} project." for item in items]
        answer = str(len(set(items)))
        question = rng.choice([
            f"How many distinct tools did I use for the completed {topic} projects?",
            f"Counting duplicates only once, how many different {topic} tools did I mention using?",
            f"What is the number of unique tools across my {topic} project updates?",
            f"After deduplicating repeated mentions, how many {topic} tools remain?",
            f"What distinct-tool count is supported across the completed {topic} work?",
            f"How many unique tool names occur in my separate {topic} project records?",
            f"If the same tool is counted once, what is the {topic} total?",
            f"What is the cardinality of the tool set used for those {topic} projects?",
        ])
    elif subtype == "average":
        statements = [
            f"The completed {topic} session lasted {value} minutes from start to finish."
            for value in values
        ]
        average = sum(values) / len(values)
        answer = f"{average:g} minutes"
        question = rng.choice([
            f"What was the average duration of the completed {topic} sessions?",
            f"On average, how many minutes did my separate {topic} sessions last?",
            f"What mean session length do the recorded {topic} durations produce?",
            f"What arithmetic mean results from the completed {topic} session lengths?",
            f"Across all recorded sessions, what was my typical {topic} duration in minutes?",
            f"If the separate durations are averaged, how long was a {topic} session?",
        ])
    elif subtype == "difference":
        first, second = sorted(values[:2] if len(values) >= 2 else [values[0], values[0] + 20])
        statements = [
            f"I paid ${first} for the train portion of the {topic} trip.",
            f"The comparable taxi for that same {topic} trip cost ${second}.",
        ]
        statements.extend(
            f"I confirmed that the {label} was part of the same comparison but carried no additional fare."
            for label in labels[2:answer_count]
        )
        answer = f"${second - first}"
        question = rng.choice([
            f"How much did I save by taking the train instead of the taxi for the {topic} trip?",
            f"What was the price difference between the taxi and train on that {topic} journey?",
            f"How much more expensive was the taxi than the train for my {topic} plans?",
            f"Subtracting the rail fare from the taxi fare, what was my {topic} saving?",
            f"What amount separates the two transport prices for the same {topic} trip?",
            f"By choosing the train, how many dollars less did I pay on the {topic} journey?",
            f"What fare gap did I record between taxi travel and rail travel for {topic}?",
            f"How much extra would the taxi have cost compared with my {topic} train fare?",
        ])
    elif subtype == "arg-extreme":
        statements = [
            f"My completed {topic} expense at {label} was ${value}."
            for label, value in zip(labels, values)
        ]
        answer = labels[values.index(max(values))]
        question = rng.choice([
            f"Which provider accounted for my largest completed {topic} expense?",
            f"Where did I spend the most money among the separate {topic} purchases?",
            f"Which named {topic} expense was the highest?",
            f"What provider corresponds to the maximum completed cost for {topic}?",
            f"Among my {topic} payments, which label carries the greatest amount?",
            f"Which completed {topic} purchase ranks first by expense?",
            f"Where is the highest confirmed charge in my {topic} records?",
            f"What named item had the largest price across those {topic} updates?",
        ])
    elif subtype == "collect-or-join":
        items = [f"{label} at {rng.choice(PLACES)}" for label in labels[:answer_count]]
        statements = [f"I completed the {topic} activity using the {item}." for item in items]
        answer = "; ".join(items)
        question = rng.choice([
            f"Which places and items were involved in my completed {topic} activities?",
            f"What were the recorded item-and-place pairs for my {topic} projects?",
            f"Can you list the completed {topic} activities with their locations?",
            f"What complete set of items and venues is supported for my {topic} work?",
            f"Which item belongs with each location across the finished {topic} activities?",
            f"Can you reconstruct every recorded {topic} item-location pairing?",
            f"What were all the completed {topic} combinations of object and place?",
            f"List the paired items and locations from my separate {topic} records.",
        ])
    else:
        statements = [
            f"The completed {topic} expense for the {label} was ${value}."
            for label, value in zip(labels, values)
        ]
        answer = f"${sum(values)}"
        question = rng.choice([
            f"How much did I spend altogether on the completed {topic} expenses?",
            f"What total results from adding all my separate {topic} costs?",
            f"Across those conversations, what was the combined amount paid for {topic}?",
            f"What aggregate payment do the completed {topic} expenses produce?",
            f"After summing each confirmed cost, how much did my {topic} activity cost?",
            f"What is the cumulative amount I paid across the separate {topic} records?",
            f"How large was my total outlay for all completed {topic} purchases?",
            f"When every recorded {topic} charge is combined, what amount results?",
            f"What final sum is supported by the individual {topic} payments?",
            f"How much money did the completed {topic} expenses add up to?",
        ])
    if profile.abstention:
        omitted = rng.choice(labels)
        question = f"What was the total completed cost of the {topic} expenses, including the {omitted}?"
        answer = INSUFFICIENT_ANSWER
        statements = [
            f"I completed the {topic} purchase for the {label} at ${value}."
            if label != omitted
            else f"I considered the {omitted} for {topic}, but I neither purchased it nor recorded a price."
            for label, value in zip(labels, values)
        ]
    question = fit_question(question, profile.question_words, rng)
    answer_text = expanded_answer(str(answer), profile.answer_words, topic)
    return Scenario(
        question, answer_text, f"multi-{subtype}", topic,
        evidence_for_profile(profile, statements, "user", "multi-session-fact"),
        [
            f"I estimated ${rng.randint(120, 400)} for a future {topic} plan, but that amount was only a budget and was never spent.",
            f"{rng.choice(PEOPLE)} completed a separate {topic} purchase that should not be counted as mine.",
        ],
    )


def temporal_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    subtype = profile.subtype
    answer_count = max(1, desired_evidence_turn_count(profile))
    answer_dates = [profile.session_dates[i] for i in profile.answer_session_indices]
    event_pool = [f"{topic_phrase} {kind}" for kind in EVENT_KINDS]
    events = rng.sample(event_pool, k=min(answer_count, len(event_pool)))
    if len(events) < answer_count:
        events.extend(f"event {i}" for i in range(len(events) + 1, answer_count + 1))
    statements = [
        f"Today I completed {event}; this was the actual event date, not the date when I first planned it."
        for event in events
    ]
    if subtype == "elapsed-duration":
        if answer_count == 1:
            gap = max(1, (profile.question_date.date() - answer_dates[0].date()).days)
            answer = f"{gap} days"
            question = rng.choice([
                f"How many days ago did I complete {events[0]}?",
                f"How many days had passed since {events[0]} by the question date?",
                f"What was the elapsed number of days after {events[0]}?",
                f"What day count separates {events[0]} from the date of this question?",
                f"By today, how many full days had elapsed since {events[0]}?",
                f"How far back in days was the completion of {events[0]}?",
                f"What interval in days lies between {events[0]} and now?",
                f"Count the days from {events[0]} through the question date; what is the gap?",
            ])
        else:
            ordered = sorted(zip(answer_dates, events))
            gap = max(0, (ordered[-1][0].date() - ordered[0][0].date()).days)
            answer = f"{gap} days"
            question = rng.choice([
                f"How many days passed between {ordered[0][1]} and {ordered[-1][1]}?",
                f"What was the elapsed time in days from {ordered[0][1]} to {ordered[-1][1]}?",
                f"How many days separated {ordered[0][1]} from {ordered[-1][1]}?",
                f"What is the day difference between {ordered[0][1]} and {ordered[-1][1]}?",
                f"From the earlier event to the later one, what was the interval in days?",
                f"How long, measured in days, was the span from {ordered[0][1]} to {ordered[-1][1]}?",
                f"What day count lies between the completion dates of {ordered[0][1]} and {ordered[-1][1]}?",
                f"Calculate the elapsed days joining {ordered[0][1]} with {ordered[-1][1]}; what is it?",
            ])
    elif subtype == "event-order":
        ordered = [event for _, event in sorted(zip(answer_dates, events))]
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        answer = "; then ".join(ordered)
        question = rng.choice([
            f"In what order did these occur, from first to last: {', '.join(shuffled)}?",
            f"Which chronological sequence is correct for {', '.join(shuffled)}?",
            f"Arrange the following events from earliest to latest: {', '.join(shuffled)}?",
            f"What is the time-ordered progression of {', '.join(shuffled)}?",
            f"Starting with the earliest, how should {', '.join(shuffled)} be sequenced?",
            f"Which event ordering follows their actual completion dates: {', '.join(shuffled)}?",
            f"Put {', '.join(shuffled)} into the order in which they happened.",
            f"Reading the timeline forward, what sequence do {', '.join(shuffled)} form?",
            f"How do these events rank from oldest to newest: {', '.join(shuffled)}?",
        ])
    elif subtype == "date-resolution":
        target = rng.randrange(answer_count)
        answer = format_timestamp(answer_dates[target]).split(" (")[0]
        question = rng.choice([
            f"On what date did I complete {events[target]}?",
            f"When did {events[target]} actually take place?",
            f"Which calendar date corresponds to {events[target]}?",
            f"What day is attached to the completion of {events[target]}?",
            f"When, according to the dated session, was {events[target]} completed?",
            f"What was the actual event date for {events[target]}?",
        ])
    else:
        latest = max(range(answer_count), key=lambda i: answer_dates[i])
        answer = events[latest]
        question = rng.choice([
            f"Which of the events I discussed happened most recently?",
            f"What was the latest completed event among the separate updates?",
            f"Which event came last according to the conversation dates?",
            f"What completed activity occupies the newest point on the timeline?",
            f"Among the dated event records, which one occurred last?",
            f"Which discussed event has the most recent completion date?",
            f"What is the final completed event when the updates are ordered by time?",
            f"Which activity was chronologically latest rather than merely planned later?",
            f"At the end of the confirmed timeline, what event appears?",
        ])
    if profile.abstention:
        unused_kinds = [kind for kind in EVENT_KINDS if f"{topic_phrase} {kind}" not in events]
        missing_event = f"{topic_phrase} {rng.choice(unused_kinds)}"
        question = f"How many days passed between {events[0]} and {missing_event}?"
        answer = INSUFFICIENT_ANSWER
        statements[-1] = f"I was considering {missing_event}, but I never scheduled or completed it and gave no date."
    question = fit_question(question, profile.question_words, rng)
    answer_text = expanded_answer(str(answer), profile.answer_words, "the event timeline")
    return Scenario(
        question, answer_text, f"temporal-{subtype}", topic_phrase,
        evidence_for_profile(profile, statements, "user", "dated-event"),
        [
            f"I first planned {events[0]} two weeks earlier, but the planning date was not the event date.",
            f"{rng.choice(PEOPLE)} attended a similar event on another day; that date belongs to them.",
        ],
    )


def build_scenario(
    profile: StructuralProfile, rng: random.Random, topic_phrase: str
) -> Scenario:
    if not profile.answer_session_indices:
        raise ValueError(f"Profile {profile.source_index} has no answer sessions")
    if profile.question_type == "single-session-user":
        return direct_user_scenario(profile, rng, topic_phrase)
    if profile.question_type == "single-session-assistant":
        return assistant_scenario(profile, rng, topic_phrase)
    if profile.question_type == "single-session-preference":
        return preference_scenario(profile, rng, topic_phrase)
    if profile.question_type == "knowledge-update":
        return knowledge_update_scenario(profile, rng, topic_phrase)
    if profile.question_type == "multi-session":
        return multi_session_scenario(profile, rng, topic_phrase)
    if profile.question_type == "temporal-reasoning":
        return temporal_scenario(profile, rng, topic_phrase)
    raise ValueError(profile.question_type)


def sentence_bank(topic_pack: tuple[str, str, str, str], role: str, rng: random.Random) -> list[str]:
    topic, object_name, place_name, detail = topic_pack
    person = rng.choice(PEOPLE)
    organization = rng.choice(ORGANIZATIONS)
    if role == "user":
        return [
            f"I have been reviewing my notes about {topic}, especially the parts concerning the {object_name} and the work near the {place_name}.",
            f"The practical issue is that my earlier plan for {topic} changed after I spoke with {person}, so I am checking each detail carefully.",
            f"One entry mentions the {detail}, while another refers to a possible future option that I have not actually completed.",
            f"I would like to keep the completed events separate from estimates, recommendations, and ideas that belong to somebody else.",
            f"The discussion with {organization} added useful background, although it did not replace the facts I recorded myself.",
            f"Could you help me compare the alternatives without treating an example as a confirmed part of my own history?",
            f"There were several numbers and names in the conversation, and only some of them described what I eventually did.",
            f"I am mentioning the surrounding context because the timing and ownership of each detail matter for this topic.",
        ]
    return [
        f"A careful approach to {topic} begins by separating confirmed information from tentative plans and general examples.",
        f"For the {object_name}, review the relevant constraints, note the condition of the item, and verify which step has actually been completed.",
        f"The setting around the {place_name} can affect scheduling, access, cost, and the sequence in which the work should be performed.",
        f"It is useful to maintain a dated record for the {detail}, because later updates may supersede an earlier recommendation.",
        f"Advice from {person} or {organization} can provide context, but it should not be mistaken for a confirmed fact about the user's actions.",
        f"A practical checklist can include preparation, verification, execution, follow-up, and a final review of the recorded result.",
        f"When several alternatives appear together, compare them on the same criteria and preserve exact names, quantities, and units.",
        f"Future possibilities should remain marked as possibilities until the user confirms that a purchase, visit, or activity occurred.",
        f"This distinction also prevents a related example from being merged with the requested person, event, or attribute.",
        f"Finally, summarize only the evidence that directly answers the question and leave unsupported details unresolved.",
    ]


def fit_content(
    target_words: int,
    bank: list[str],
    rng: random.Random,
    required_text: str | None = None,
) -> str:
    def filler(exact_words: int) -> str:
        if exact_words <= 0:
            return ""
        pieces: list[str] = []
        current = 0
        while current < exact_words:
            sentence = rng.choice(bank)
            pieces.append(sentence)
            current += word_count(sentence)
        text = " ".join(pieces)
        token_matches = list(WORD_RE.finditer(text))
        if len(token_matches) <= exact_words:
            return text
        end = token_matches[exact_words - 1].end()
        clipped = text[:end].rstrip(" ,;:")
        return clipped + ("." if not clipped.endswith((".", "?", "!")) else "")

    if required_text:
        required = required_text.strip()
        required_words = word_count(required)
        target = max(target_words, required_words + 12)
        available = target - required_words
        prefix_words = min(max(5, target // 3), max(5, available // 2))
        suffix_words = max(0, available - prefix_words)
        prefix = filler(prefix_words)
        suffix = filler(suffix_words)
        return " ".join(part for part in [prefix, required, suffix] if part)

    target = target_words
    pieces: list[str] = []
    current = 0
    while current < target:
        sentence = rng.choice(bank)
        pieces.append(sentence)
        current += word_count(sentence)
    text = " ".join(pieces)
    token_matches = list(WORD_RE.finditer(text))
    if len(token_matches) <= target:
        return text
    end = token_matches[target - 1].end()
    clipped = text[:end].rstrip(" ,;:")
    return clipped + ("." if not clipped.endswith((".", "?", "!")) else "")


def allocate_targets_with_minimum(total: int, minimums: list[int]) -> list[int]:
    if not minimums:
        return []
    total = max(int(total), sum(minimums))
    extras = distribute_integer_total(total - sum(minimums), len(minimums))
    return [minimum + extra for minimum, extra in zip(minimums, extras)]


def evidence_targets_for_scenario(
    scenario: Scenario,
    aggregate_target: RelevantEvidenceTarget | None,
) -> dict[int, deque[RelevantEvidenceTarget]]:
    """Allocate one aggregate density target across provenance-bearing evidence."""
    evidence = list(scenario.evidence)
    if not evidence or aggregate_target is None:
        return {}
    word_targets = allocate_targets_with_minimum(
        aggregate_target.words,
        [word_count(item.text) for item in evidence],
    )
    sentence_targets = allocate_targets_with_minimum(
        aggregate_target.sentences,
        [sentence_count(item.text) for item in evidence],
    )
    numeric_targets = allocate_targets_with_minimum(
        aggregate_target.numeric_mentions,
        [numeric_mention_count(item.text) for item in evidence],
    )
    targets: dict[int, deque[RelevantEvidenceTarget]] = defaultdict(deque)
    for index, item in enumerate(evidence):
        targets[item.answer_ordinal].append(RelevantEvidenceTarget(
            words=word_targets[index],
            sentences=sentence_targets[index],
            numeric_mentions=numeric_targets[index],
        ))
    return targets


def controlled_filler_sentence(
    role: str,
    topic: str,
    target_words: int,
    numeric_mentions: int,
    rng: random.Random,
) -> str:
    """Create one natural surrounding sentence with an exact token/number budget."""
    number_nouns = [
        "drafts", "examples", "options", "stages", "notes", "checks",
        "minutes", "items", "sections", "steps", "estimates", "entries",
    ]
    numbers = [str(rng.randint(2, 96)) for _ in range(numeric_mentions)]
    if numbers:
        tokens = list(rng.choice([
            ["The", "supporting", "notes", "recorded"],
            ["Another", "record", "section", "compared"],
            ["The", "earlier", "discussion", "tracked"],
            ["Background", "notes", "carefully", "distinguished"],
        ]))
        for index, number in enumerate(numbers):
            if index:
                tokens.append("and")
            tokens.extend([number, rng.choice(number_nouns)])
    elif role == "assistant":
        tokens = list(rng.choice([
            ["The", "explanation", "also", "reviewed", "practical"],
            ["Another", "part", "of", "the", "response"],
            ["The", "earlier", "guidance", "also", "considered"],
            ["The", "answer", "also", "separated", "related"],
        ]))
    else:
        tokens = list(rng.choice([
            ["I", "also", "described", "practical"],
            ["My", "notes", "covered", "related"],
            ["The", "conversation", "recorded", "background"],
            ["I", "also", "considered", "separate"],
        ]))

    # A numeric topic token (for example, "17-year") belongs in the source
    # fact, but repeating it in every filler sentence artificially multiplies
    # the turn's numeric density.
    topic_tokens = [
        value.lower()
        for value in words(topic)
        if not any(character.isdigit() for character in value)
    ][:3]
    tokens.extend(topic_tokens or ["this", "topic"])
    tokens.extend(
        ["as", "context"]
        if role == "assistant"
        else ["in", "my", "notes"]
    )
    minimum = len(tokens)
    target_words = max(int(target_words), minimum)
    padding = [
        "carefully", "separately", "earlier", "later", "practical", "related",
        "background", "planning", "review", "comparison", "schedule", "summary",
        "follow-up", "discussion", "detail", "example", "alternative", "record",
        "context", "confirmed", "tentative", "specific", "additional", "useful",
    ]
    rng.shuffle(padding)
    while len(tokens) < target_words:
        tokens.append(padding[(len(tokens) - minimum) % len(padding)])
    return " ".join(tokens) + "."


def natural_filler_sentences(
    background_sentences: list[str],
    word_allocations: list[int],
    number_allocations: list[int],
    rng: random.Random,
) -> list[str]:
    """Reshape independent prose while preserving exact density targets."""
    replacements = [
        "several", "various", "another", "additional", "earlier", "later",
        "multiple", "separate", "different", "further", "specific", "recent",
    ]
    result = []
    extensions = [
        ["with", "the", "remaining", "context", "kept", "in", "the", "same", "discussion"],
        ["while", "the", "surrounding", "explanation", "continued", "with", "related", "details"],
        ["and", "the", "broader", "account", "retained", "its", "practical", "background"],
        ["as", "the", "conversation", "moved", "through", "the", "supporting", "points"],
    ]
    for index, (word_target, number_target) in enumerate(
        zip(word_allocations, number_allocations)
    ):
        source = background_sentences[index] if index < len(background_sentences) else ""
        segment = [
            rng.choice(replacements)
            if any(character.isdigit() for character in token)
            else token
            for token in words(source)
        ][:word_target]
        extension = extensions[index % len(extensions)]
        while len(segment) < word_target:
            segment.append(extension[(len(segment) - len(words(source))) % len(extension)])
        for number_index in range(min(number_target, len(segment))):
            position = len(segment) - number_index - 1
            segment[position] = str(rng.randint(2, 96))
        if segment:
            segment[0] = segment[0][0].upper() + segment[0][1:]
        result.append(" ".join(segment) + ".")
    return result


def evidence_filler_plan(
    required_text: str,
    target: RelevantEvidenceTarget,
) -> tuple[list[int], list[int]]:
    core_words = word_count(required_text)
    core_sentences = sentence_count(required_text)
    core_numbers = numeric_mention_count(required_text)
    missing_numbers = max(0, target.numeric_mentions - core_numbers)
    filler_count = max(0, target.sentences - core_sentences)
    if filler_count == 0 and (
        target.words > core_words or missing_numbers > 0
    ):
        filler_count = 1
    number_allocations = distribute_integer_total(missing_numbers, filler_count)
    minimum_filler_words = [
        max(8, 10 + 3 * count)
        for count in number_allocations
    ]
    filler_word_allocations = allocate_targets_with_minimum(
        max(0, target.words - core_words),
        minimum_filler_words,
    )
    return number_allocations, filler_word_allocations


def compose_calibrated_evidence_turn(
    required_text: str,
    target: RelevantEvidenceTarget,
    role: str,
    topic: str,
    rng: random.Random,
    background_sentences: list[str] | None = None,
) -> str:
    """Preserve the source fact while matching full-turn evidence complexity."""
    required = required_text.strip()
    number_allocations, filler_word_allocations = evidence_filler_plan(
        required,
        target,
    )
    filler_count = len(filler_word_allocations)
    if background_sentences:
        fillers = natural_filler_sentences(
            background_sentences,
            filler_word_allocations,
            number_allocations,
            rng,
        )
    else:
        fillers = [
            controlled_filler_sentence(
                role=role,
                topic=topic,
                target_words=filler_word_allocations[index],
                numeric_mentions=number_allocations[index],
                rng=rng,
            )
            for index in range(filler_count)
        ]
    if not fillers:
        return required

    insertion = rng.randint(0, len(fillers))
    parts = fillers[:insertion] + [required] + fillers[insertion:]
    return " ".join(part for part in parts if part)


def validate_evidence_preservation(record: dict, audit: dict) -> None:
    sessions = {
        session_id: session
        for session_id, session in zip(
            record["haystack_session_ids"], record["haystack_sessions"]
        )
    }
    for evidence in audit["evidence"]:
        session = sessions.get(evidence["session_id"])
        turn_index = evidence["turn_index"]
        if session is None or turn_index >= len(session):
            raise ValueError(
                f"Invalid evidence location for {record['question_id']}: {evidence}"
            )
        if evidence["source_text"] not in session[turn_index]["content"]:
            raise ValueError(
                f"Evidence text was not preserved for {record['question_id']} "
                f"at {evidence['session_id']}:{turn_index}"
            )


def neutralize_exact_official_turn_overlaps(
    record: dict,
    audit: dict,
    official_turn_hashes: set[str],
) -> int:
    """Prevent split background fragments from becoming exact official turns."""
    evidence_locations = {
        (evidence["session_id"], int(evidence["turn_index"]))
        for evidence in audit.get("evidence") or []
    }
    neutralized = 0
    for session_id, session in zip(
        record["haystack_session_ids"], record["haystack_sessions"]
    ):
        for turn_index, message in enumerate(session):
            content = str(message.get("content") or "")
            if word_count(content) < 5:
                continue
            if normalized_text_hash(content) not in official_turn_hashes:
                continue
            if (session_id, turn_index) in evidence_locations:
                raise ValueError(
                    "Generated evidence unexpectedly equals an official turn: "
                    f"{record['question_id']} {session_id}:{turn_index}"
                )
            message["content"] = (
                content
                + " This separate background note remains unrelated to the requested fact."
            )
            neutralized += 1
    return neutralized


def anchor_coherent_distractor(
    question: str,
    answer: object,
    topic: str,
    distractor: str,
    rng: random.Random,
) -> str:
    """Add question anchors to a false but topically plausible evidence turn."""
    answer_tokens = {value.lower() for value in words(answer)}
    candidates = []
    seen = set()
    for token in words(question):
        normalized = token.lower()
        if (
            normalized in QUERY_ANCHOR_STOPWORDS
            or normalized in answer_tokens
            or any(character.isdigit() for character in normalized)
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        candidates.append(normalized)
    rng.shuffle(candidates)
    anchors = candidates[: min(4, max(2, len(candidates) // 2))]
    if not anchors:
        return distractor
    if len(anchors) == 1:
        anchor_phrase = anchors[0]
    else:
        anchor_phrase = ", ".join(anchors[:-1]) + f", and {anchors[-1]}"
    lead = rng.choice([
        f"The {topic} notes also discussed {anchor_phrase}.",
        f"Another {topic} exchange connected {anchor_phrase}.",
        f"A nearby {topic} entry covered {anchor_phrase}.",
    ])
    return f"{lead} {distractor}"


def map_session_dates(profile: StructuralProfile, synthetic_question_date: datetime) -> list[datetime]:
    return [
        synthetic_question_date - (profile.question_date - session_date)
        for session_date in profile.session_dates
    ]


def build_record(
    profile: StructuralProfile,
    seed: int,
    background: BackgroundProvider | None = None,
    topic_phrase: str | None = None,
) -> tuple[dict, dict]:
    rng = random.Random(stable_record_seed(seed, profile.source_index))
    scenario = build_scenario(
        profile, rng, topic_phrase or rng.choice(ACTIVITIES)
    )
    if profile.suppress_relevant_evidence:
        scenario = Scenario(
            question=scenario.question,
            answer=scenario.answer,
            family=scenario.family,
            topic=scenario.topic,
            evidence=[],
            near_distractors=scenario.near_distractors,
        )
    synthetic_question_date = datetime(2025, 1, 15, 20, 0) + timedelta(
        days=profile.source_index * 3,
        minutes=rng.randint(0, 360),
    )
    session_dates = map_session_dates(profile, synthetic_question_date)
    answer_index_to_ordinal = {
        session_index: ordinal
        for ordinal, session_index in enumerate(profile.answer_session_indices)
    }
    evidence_by_ordinal = defaultdict(list)
    for evidence in scenario.evidence:
        evidence_by_ordinal[evidence.answer_ordinal].append(evidence)
    evidence_density_targets = evidence_targets_for_scenario(
        scenario,
        profile.relevant_evidence_target,
    )

    sessions: list[list[dict]] = []
    session_ids: list[str] = []
    answer_session_ids: list[str] = []
    evidence_audit: list[dict] = []
    near_session_candidates = [
        index for index in range(len(profile.session_roles))
        if index not in answer_index_to_ordinal
    ]
    near_session_indices = set(rng.sample(
        near_session_candidates,
        k=min(len(near_session_candidates), max(3, len(profile.answer_session_indices) * 2)),
    ))
    hardening_seed = stable_record_seed(seed ^ 0x5E6A71C, profile.source_index)
    hardening_rate = HARD_DISTRACTOR_RATES[profile.question_type]
    harden_record = (hardening_seed % 10_000) / 10_000 < hardening_rate
    hardened_session_index = None
    if harden_record and near_session_indices:
        ordered_near = sorted(near_session_indices)
        hardened_session_index = ordered_near[hardening_seed % len(ordered_near)]

    for session_index, (roles, counts) in enumerate(
        zip(profile.session_roles, profile.session_word_counts)
    ):
        is_answer = session_index in answer_index_to_ordinal
        answer_ordinal = answer_index_to_ordinal.get(session_index)
        session_id = (
            f"synv2_{profile.source_index:04d}_answer_{answer_ordinal + 1}"
            if is_answer
            else f"synv2_{profile.source_index:04d}_noise_{session_index + 1}"
        )
        session_ids.append(session_id)
        if is_answer:
            answer_session_ids.append(session_id)

        topic_pack = (
            (scenario.topic, "record", rng.choice(PLACES), "supporting detail")
            if is_answer or session_index in near_session_indices
            else rng.choice(TOPIC_PACKS)
        )
        messages = []
        pending = list(evidence_by_ordinal.get(answer_ordinal, [])) if is_answer else []
        evidence_turns: list[int] = []
        eligible_turns = [
            i for i, role in enumerate(roles)
            if not pending or role == pending[0].role
        ]
        if pending and not eligible_turns:
            eligible_turns = list(range(len(roles)))
        chosen_turns = []
        if pending:
            lower = max(0, len(eligible_turns) // 3)
            pool = eligible_turns[lower:] or eligible_turns
            chosen_turns = sorted(rng.sample(pool, k=min(len(pool), len(pending))))
            while len(chosen_turns) < len(pending):
                chosen_turns.append(chosen_turns[-1] if chosen_turns else 0)

        for turn_index, (role, target_words) in enumerate(zip(roles, counts)):
            required = None
            evidence_turn_target = None
            if pending and turn_index in chosen_turns:
                evidence = pending.pop(0)
                required = evidence.text
                role = evidence.role
                target_queue = evidence_density_targets.get(evidence.answer_ordinal)
                evidence_turn_target = (
                    target_queue.popleft() if target_queue else None
                )
                evidence_turns.append(turn_index)
                evidence_audit.append({
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "role": role,
                    "evidence_type": evidence.evidence_type,
                    "source_text": evidence.text,
                    "density_target": (
                        {
                            "words": evidence_turn_target.words,
                            "sentences": evidence_turn_target.sentences,
                            "numeric_mentions": evidence_turn_target.numeric_mentions,
                        }
                        if evidence_turn_target is not None
                        else None
                    ),
                })
            elif session_index in near_session_indices and scenario.near_distractors:
                if turn_index == max(0, len(roles) // 2):
                    distractor = scenario.near_distractors[
                        session_index % len(scenario.near_distractors)
                    ]
                    required = (
                        anchor_coherent_distractor(
                            scenario.question,
                            scenario.answer,
                            scenario.topic,
                            distractor,
                            rng,
                        )
                        if session_index == hardened_session_index
                        else distractor
                    )
            bank = sentence_bank(topic_pack, role, rng)
            if required is not None and evidence_turn_target is not None:
                _, filler_word_targets = evidence_filler_plan(
                    required,
                    evidence_turn_target,
                )
                background_sentences = (
                    background.take_topic_sentence_fragments(
                        scenario.topic,
                        role,
                        filler_word_targets,
                    )
                    if background is not None
                    else None
                )
                content = compose_calibrated_evidence_turn(
                    required_text=required,
                    target=evidence_turn_target,
                    role=role,
                    topic=scenario.topic,
                    rng=rng,
                    background_sentences=background_sentences,
                )
            else:
                content = (
                    background.take(role, target_words, required)
                    if background is not None
                    else fit_content(target_words, bank, rng, required)
                )
            messages.append({
                "role": role,
                "content": content,
            })
        if pending:
            for evidence in pending:
                turn_index = len(messages) - 1
                messages[turn_index]["role"] = evidence.role
                messages[turn_index]["content"] += " " + evidence.text
                evidence_audit.append({
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "role": evidence.role,
                    "evidence_type": evidence.evidence_type,
                    "source_text": evidence.text,
                })
                evidence_turns.append(turn_index)
        sessions.append(messages)

    suffix = "_abs" if profile.abstention else ""
    question_id = f"synthetic_v2_{profile.source_index:04d}{suffix}"
    record = {
        "question_id": question_id,
        "question_type": profile.question_type,
        "question": scenario.question,
        "answer": scenario.answer,
        "question_date": format_timestamp(synthetic_question_date),
        "answer_session_ids": answer_session_ids,
        "haystack_session_ids": session_ids,
        "haystack_dates": [format_timestamp(value) for value in session_dates],
        "haystack_sessions": sessions,
    }
    audit = {
        "question_id": question_id,
        "source_profile_index": profile.source_index,
        "question_type": profile.question_type,
        "subtype": profile.subtype,
        "family": scenario.family,
        "abstention": profile.abstention,
        "evidence": evidence_audit,
        "near_distractor_session_ids": [session_ids[i] for i in sorted(near_session_indices)],
        "official_shape": {
            "session_count": len(profile.session_roles),
            "turn_count": sum(map(len, profile.session_roles)),
            "answer_session_count": len(profile.answer_session_indices),
            "question_words": profile.question_words,
            "answer_words": profile.answer_words,
        },
    }
    return record, audit


class DistributionAccumulator:
    def __init__(self) -> None:
        self.records = 0
        self.question_types: Counter[str] = Counter()
        self.subtypes: Counter[str] = Counter()
        self.abstentions = 0
        self.session_counts: list[int] = []
        self.turn_counts: list[int] = []
        self.source_words: list[int] = []
        self.gold_session_words: list[int] = []
        self.question_words: list[int] = []
        self.answer_words: list[int] = []
        self.answer_session_counts: list[int] = []
        self.role_words: dict[str, list[int]] = defaultdict(list)
        self.record_ttr: list[float] = []
        self.unique_turn_rate: list[float] = []
        self.trigram_diversity: list[float] = []
        self.corpus_question_tokens: list[str] = []

    def add(self, record: dict, subtype: str) -> None:
        self.records += 1
        self.question_types[record["question_type"]] += 1
        self.subtypes[f"{record['question_type']}::{subtype}"] += 1
        self.abstentions += int("_abs" in record["question_id"])
        sessions = record["haystack_sessions"]
        ids = record["haystack_session_ids"]
        gold = set(record["answer_session_ids"])
        self.session_counts.append(len(sessions))
        self.turn_counts.append(sum(len(session) for session in sessions))
        self.answer_session_counts.append(len(gold))
        source_total = 0
        gold_total = 0
        record_tokens: list[str] = []
        turn_texts: list[str] = []
        for session_id, session in zip(ids, sessions):
            for message in session:
                content = str(message.get("content") or "")
                tokens = [value.lower() for value in words(content)]
                count = len(tokens)
                source_total += count
                record_tokens.extend(tokens)
                turn_texts.append(" ".join(content.lower().split()))
                self.role_words[str(message.get("role"))].append(count)
                if session_id in gold:
                    gold_total += count
        self.source_words.append(source_total)
        self.gold_session_words.append(gold_total)
        self.question_words.append(word_count(record.get("question")))
        self.corpus_question_tokens.extend(
            value.lower() for value in words(record.get("question"))
        )
        self.answer_words.append(word_count(record.get("answer")))
        self.record_ttr.append(
            len(set(record_tokens)) / len(record_tokens) if record_tokens else 0.0
        )
        self.unique_turn_rate.append(
            len(set(turn_texts)) / len(turn_texts) if turn_texts else 0.0
        )
        trigrams = list(zip(record_tokens, record_tokens[1:], record_tokens[2:]))
        self.trigram_diversity.append(
            len(set(trigrams)) / len(trigrams) if trigrams else 0.0
        )

    @staticmethod
    def _stats(values: list[int]) -> dict:
        ordered = sorted(values)
        if not ordered:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(ordered),
            "median": statistics.median(ordered),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "min": ordered[0],
            "max": ordered[-1],
        }

    def summary(self) -> dict:
        question_trigrams = list(zip(
            self.corpus_question_tokens,
            self.corpus_question_tokens[1:],
            self.corpus_question_tokens[2:],
        ))
        return {
            "records": self.records,
            "question_type_counts": dict(sorted(self.question_types.items())),
            "subtype_counts": dict(sorted(self.subtypes.items())),
            "abstention_count": self.abstentions,
            "session_count": self._stats(self.session_counts),
            "turn_count": self._stats(self.turn_counts),
            "source_words": self._stats(self.source_words),
            "gold_session_words": self._stats(self.gold_session_words),
            "question_words": self._stats(self.question_words),
            "answer_words": self._stats(self.answer_words),
            "answer_session_count": self._stats(self.answer_session_counts),
            "record_ttr": self._stats(self.record_ttr),
            "unique_turn_rate": self._stats(self.unique_turn_rate),
            "trigram_diversity": self._stats(self.trigram_diversity),
            "question_lexical_diversity": {
                "token_count": len(self.corpus_question_tokens),
                "vocabulary_size": len(set(self.corpus_question_tokens)),
                "ttr": (
                    len(set(self.corpus_question_tokens)) / len(self.corpus_question_tokens)
                    if self.corpus_question_tokens else 0.0
                ),
                "trigram_diversity": (
                    len(set(question_trigrams)) / len(question_trigrams)
                    if question_trigrams else 0.0
                ),
            },
            "role_words": {
                role: self._stats(values) for role, values in sorted(self.role_words.items())
            },
        }


def summarize_official(records: list[dict]) -> dict:
    accumulator = DistributionAccumulator()
    for index, record in enumerate(records):
        accumulator.add(record, question_subtype(record))
    return accumulator.summary()


def relative_delta(candidate: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if candidate == 0 else math.inf
    return (candidate - reference) / reference


def distribution_comparison(official: dict, synthetic: dict) -> dict:
    metrics = {}
    for section in [
        "session_count", "turn_count", "source_words", "gold_session_words",
        "question_words", "answer_words", "answer_session_count",
        "record_ttr", "unique_turn_rate", "trigram_diversity",
    ]:
        metrics[section] = {
            key: {
                "official": official[section][key],
                "synthetic": synthetic[section][key],
                "relative_delta": relative_delta(
                    synthetic[section][key], official[section][key]
                ),
            }
            for key in ["mean", "median", "p90"]
        }
    checks = {
        "record_count_exact": synthetic["records"] == official["records"] == 500,
        "question_type_counts_exact": synthetic["question_type_counts"] == official["question_type_counts"],
        "subtype_counts_exact": synthetic["subtype_counts"] == official["subtype_counts"],
        "abstention_count_exact": synthetic["abstention_count"] == official["abstention_count"],
        "session_mean_within_1pct": abs(metrics["session_count"]["mean"]["relative_delta"]) <= 0.01,
        "turn_mean_within_1pct": abs(metrics["turn_count"]["mean"]["relative_delta"]) <= 0.01,
        "source_words_mean_within_8pct": abs(metrics["source_words"]["mean"]["relative_delta"]) <= 0.08,
        "gold_words_mean_within_10pct": abs(metrics["gold_session_words"]["mean"]["relative_delta"]) <= 0.10,
        "question_words_mean_within_12pct": abs(metrics["question_words"]["mean"]["relative_delta"]) <= 0.12,
        "answer_words_mean_within_10pct": abs(metrics["answer_words"]["mean"]["relative_delta"]) <= 0.10,
        "answer_words_p90_within_15pct": abs(metrics["answer_words"]["p90"]["relative_delta"]) <= 0.15,
        "answer_sessions_mean_within_1pct": abs(metrics["answer_session_count"]["mean"]["relative_delta"]) <= 0.01,
        "record_ttr_mean_within_20pct": abs(metrics["record_ttr"]["mean"]["relative_delta"]) <= 0.20,
        "unique_turn_rate_mean_within_2pct": abs(metrics["unique_turn_rate"]["mean"]["relative_delta"]) <= 0.02,
        "trigram_diversity_mean_within_20pct": abs(metrics["trigram_diversity"]["mean"]["relative_delta"]) <= 0.20,
        "question_ttr_within_30pct": abs(relative_delta(
            synthetic["question_lexical_diversity"]["ttr"],
            official["question_lexical_diversity"]["ttr"],
        )) <= 0.30,
        "question_trigram_diversity_within_25pct": abs(relative_delta(
            synthetic["question_lexical_diversity"]["trigram_diversity"],
            official["question_lexical_diversity"]["trigram_diversity"],
        )) <= 0.25,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": metrics,
        "question_lexical_diversity": {
            "official": official["question_lexical_diversity"],
            "synthetic": synthetic["question_lexical_diversity"],
        },
    }


def relevant_evidence_comparison(official: dict, synthetic: dict) -> dict:
    def compare_section(reference: dict, candidate: dict) -> dict:
        return {
            metric: {
                "official": reference[metric]["mean"],
                "synthetic": candidate[metric]["mean"],
                "relative_delta": relative_delta(
                    candidate[metric]["mean"], reference[metric]["mean"]
                ),
            }
            for metric in ["words", "sentences", "numeric_mentions"]
        }

    overall = compare_section(official["overall"], synthetic["overall"])
    by_question_type = {
        question_type: compare_section(
            official["by_question_type"][question_type],
            synthetic["by_question_type"][question_type],
        )
        for question_type in sorted(official["by_question_type"])
    }
    assistant = by_question_type["single-session-assistant"]
    checks = {
        "relevant_words_mean_within_5pct": abs(
            overall["words"]["relative_delta"]
        ) <= 0.05,
        "relevant_sentences_mean_within_5pct": abs(
            overall["sentences"]["relative_delta"]
        ) <= 0.05,
        "relevant_numeric_mentions_mean_within_8pct": abs(
            overall["numeric_mentions"]["relative_delta"]
        ) <= 0.08,
        "assistant_relevant_words_mean_within_5pct": abs(
            assistant["words"]["relative_delta"]
        ) <= 0.05,
        "assistant_numeric_mentions_mean_within_8pct": abs(
            assistant["numeric_mentions"]["relative_delta"]
        ) <= 0.08,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "overall": overall,
        "by_question_type": by_question_type,
    }


def write_dataset(
    official_records: list[dict],
    relevant_evidence_targets: dict[int, RelevantEvidenceTarget],
    relevant_turn_counts: dict[int, int],
    zero_evidence_indices: set[int],
    output_path: Path,
    audit_path: Path,
    seed: int,
    background: BackgroundProvider,
    topic_phrases: list[str],
) -> tuple[dict, dict, dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_temp = output_path.with_suffix(output_path.suffix + ".tmp")
    audit_temp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    accumulator = DistributionAccumulator()
    generated_questions: set[str] = set()
    official_questions = {str(record.get("question") or "").strip().lower() for record in official_records}
    exact_question_overlap = 0
    long_answer_overlap = 0
    official_long_answers = {
        str(record.get("answer") or "").strip().lower()
        for record in official_records if word_count(record.get("answer")) >= 6
    }
    official_long_turn_hashes = {
        normalized_text_hash(message.get("content"))
        for record in official_records
        for session in record.get("haystack_sessions") or []
        for message in session
        if word_count(message.get("content")) >= 5
    }
    exact_long_turn_overlap = 0
    neutralized_turn_overlap = 0
    relevant_evidence_rows: list[
        tuple[str, RelevantEvidenceTarget, int]
    ] = []

    with dataset_temp.open("w", encoding="utf-8") as dataset_handle, audit_temp.open(
        "w", encoding="utf-8"
    ) as audit_handle:
        dataset_handle.write("[\n")
        for index, official_record in enumerate(official_records):
            profile = build_profile(
                official_record,
                index,
                relevant_evidence_target=relevant_evidence_targets.get(index),
                target_relevant_turn_count=relevant_turn_counts[index],
                suppress_relevant_evidence=index in zero_evidence_indices,
            )
            record, audit = build_record(
                profile,
                seed,
                background,
                topic_phrases[index % len(topic_phrases)],
            )
            neutralized_turn_overlap += neutralize_exact_official_turn_overlaps(
                record,
                audit,
                official_long_turn_hashes,
            )
            validate_evidence_preservation(record, audit)
            relevant_metrics, relevant_turn_count = synthetic_relevant_evidence_metrics(
                record,
                audit,
            )
            relevant_evidence_rows.append(
                (record["question_type"], relevant_metrics, relevant_turn_count)
            )
            normalized_question = record["question"].strip().lower()
            exact_question_overlap += int(normalized_question in official_questions)
            long_answer_overlap += int(
                word_count(record["answer"]) >= 6
                and str(record["answer"]).strip().lower() in official_long_answers
            )
            exact_long_turn_overlap += sum(
                normalized_text_hash(message.get("content")) in official_long_turn_hashes
                for session in record["haystack_sessions"]
                for message in session
                if word_count(message.get("content")) >= 5
            )
            if normalized_question in generated_questions:
                record["question"] = record["question"].rstrip("?") + f" for record {index + 1}?"
                normalized_question = record["question"].strip().lower()
            generated_questions.add(normalized_question)
            accumulator.add(record, profile.subtype)
            if index:
                dataset_handle.write(",\n")
            json.dump(record, dataset_handle, ensure_ascii=True, separators=(",", ":"))
            audit_handle.write(json.dumps(audit, ensure_ascii=True, separators=(",", ":")) + "\n")
        dataset_handle.write("\n]\n")
    dataset_temp.replace(output_path)
    audit_temp.replace(audit_path)
    leakage = {
        "exact_official_question_overlap_count": exact_question_overlap,
        "exact_official_long_answer_overlap_count": long_answer_overlap,
        "unique_generated_question_count": len(generated_questions),
        "exact_official_turn_overlap_count_min_5_words": exact_long_turn_overlap,
        "neutralized_background_turn_overlap_count": neutralized_turn_overlap,
    }
    return (
        accumulator.summary(),
        leakage,
        summarize_relevant_metric_rows(relevant_evidence_rows),
    )


def build_manifest(
    official_path: Path,
    official_oracle_path: Path,
    output_path: Path,
    audit_path: Path,
    seed: int,
    official_summary: dict,
    synthetic_summary: dict,
    comparison: dict,
    official_relevant_evidence: dict,
    synthetic_relevant_evidence: dict,
    relevant_evidence_comparison_result: dict,
    leakage: dict,
    background_manifest_path: Path,
    topic_context_summary: dict,
) -> dict:
    background_manifest = json.loads(
        background_manifest_path.read_text(encoding="utf-8")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "PRAGMOS Synthetic LongMemEval Development Set V2",
        "purpose": "high_fidelity_development_and_regression_testing_only",
        "official_longmemeval_score": False,
        "independent_evaluation_set": False,
        "seed": seed,
        "generator": Path(__file__).name,
        "generator_sha256": sha256_path(Path(__file__)),
        "official_profile_source": str(official_path),
        "official_profile_source_sha256": sha256_path(official_path),
        "official_evidence_profile_source": str(official_oracle_path),
        "official_evidence_profile_source_sha256": sha256_path(
            official_oracle_path
        ),
        "dataset_file": output_path.name,
        "dataset_sha256": sha256_path(output_path),
        "audit_file": audit_path.name,
        "audit_sha256": sha256_path(audit_path),
        "audit_warning": "The audit file contains evidence locations and must never be supplied to inference.",
        "background_corpus": {
            "manifest_path": str(background_manifest_path),
            "manifest_sha256": sha256_path(background_manifest_path),
            "source_dataset": background_manifest["source_dataset"],
            "source_revision": background_manifest["source_revision"],
            "source_license": background_manifest["source_license"],
            "rejected_exact_official_turn_overlap_count": background_manifest[
                "rejected_exact_official_turn_overlap_count"
            ],
            "files": background_manifest["files"],
            "topic_phrases": background_manifest["topic_phrases"],
            "topic_context_index": topic_context_summary,
        },
        "profile_fields_used": [
            "question_type", "question text for anonymous subtype classification",
            "question_id for abstention classification", "question/answer word counts",
            "question/session timestamps", "answer-session positions",
            "per-session role patterns", "per-turn word counts capped at 900",
            "per-question-type aggregate answer-bearing turn word counts",
            "per-question-type aggregate answer-bearing sentence counts",
            "per-question-type aggregate answer-bearing numeric-mention counts",
            "per-record answer-bearing-turn count",
        ],
        "content_never_emitted": [
            "official questions", "official answers", "official entities",
            "official conversation text", "official session identifiers",
        ],
        "official_distribution": official_summary,
        "synthetic_distribution": synthetic_summary,
        "distribution_comparison": comparison,
        "official_relevant_evidence_distribution": official_relevant_evidence,
        "synthetic_relevant_evidence_distribution": synthetic_relevant_evidence,
        "relevant_evidence_comparison": relevant_evidence_comparison_result,
        "leakage_checks": leakage,
        "publication_note": (
            "V2 is benchmark-shaped development data derived from anonymous structural "
            "profiles of the official test set. It must not be presented as independent "
            "validation or as an official LongMemEval result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-data",
        default="benchmark/longMemEval/longmemeval_s_cleaned.json",
        type=Path,
    )
    parser.add_argument(
        "--official-oracle",
        default="benchmark/longMemEval/longmemeval_oracle.json",
        type=Path,
        help=(
            "Official oracle used only for aggregate answer-bearing turn density; "
            "its text is never emitted."
        ),
    )
    parser.add_argument(
        "--output",
        default="benchmark/longMemEval/synthetic_v2/pragmos_synthetic_longmemeval_500_v2.json",
        type=Path,
    )
    parser.add_argument(
        "--audit-output",
        default="benchmark/longMemEval/synthetic_v2/pragmos_synthetic_longmemeval_500_v2_audit.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--manifest-output",
        default="benchmark/longMemEval/synthetic_v2/pragmos_synthetic_longmemeval_500_v2_manifest.json",
        type=Path,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--background-user",
        type=Path,
        default=Path("benchmark/longMemEval/synthetic_v2/background/ultrachat_user_background.jsonl"),
    )
    parser.add_argument(
        "--background-assistant",
        type=Path,
        default=Path("benchmark/longMemEval/synthetic_v2/background/ultrachat_assistant_background.jsonl"),
    )
    parser.add_argument(
        "--background-manifest",
        type=Path,
        default=Path("benchmark/longMemEval/synthetic_v2/background/ultrachat_background_manifest.json"),
    )
    parser.add_argument(
        "--background-topics",
        type=Path,
        default=Path("benchmark/longMemEval/synthetic_v2/background/ultrachat_topic_phrases.json"),
    )
    args = parser.parse_args()

    official_records = json.loads(args.official_data.read_text(encoding="utf-8"))
    if len(official_records) != 500:
        raise ValueError(f"Expected 500 official records, found {len(official_records)}")
    official_summary = summarize_official(official_records)
    official_oracle_records = json.loads(
        args.official_oracle.read_text(encoding="utf-8")
    )
    if len(official_oracle_records) != len(official_records):
        raise ValueError(
            "Official oracle and cleaned dataset must contain the same record count"
        )
    official_ids = {
        str(record.get("question_id")) for record in official_records
    }
    oracle_ids = {
        str(record.get("question_id")) for record in official_oracle_records
    }
    if oracle_ids != official_ids:
        raise ValueError("Official oracle and cleaned dataset question IDs differ")
    official_relevant_evidence = official_relevant_evidence_summary(
        official_oracle_records
    )
    oracle_by_id = {
        str(record.get("question_id")): record
        for record in official_oracle_records
    }
    relevant_turn_counts = {
        index: sum(
            message.get("has_answer") is True
            for session in oracle_by_id[str(record.get("question_id"))].get(
                "haystack_sessions"
            ) or []
            for message in session
        )
        for index, record in enumerate(official_records)
    }
    zero_evidence_indices = {
        index
        for index, turn_count in relevant_turn_counts.items()
        if turn_count == 0
    }
    for path in [
        args.background_user,
        args.background_assistant,
        args.background_manifest,
        args.background_topics,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing background corpus artifact: {path}. Run "
                "prepare_PRAGMOS_synthetic_LongMemEval_v2_background.py first."
            )
    topic_phrases = json.loads(args.background_topics.read_text(encoding="utf-8"))
    if len(topic_phrases) < len(official_records):
        raise ValueError("Background topic lexicon has fewer than 500 entries")
    random.Random(args.seed).shuffle(topic_phrases)
    relevant_evidence_targets = build_relevant_evidence_targets(
        official_records,
        official_relevant_evidence,
        args.seed,
        topic_phrases,
        relevant_turn_counts,
        zero_evidence_indices,
    )
    background = BackgroundProvider(args.background_user, args.background_assistant)
    background.index_topic_contexts(topic_phrases[: len(official_records)])
    topic_context_summary = background.topic_context_summary(
        topic_phrases[: len(official_records)]
    )
    if topic_context_summary["topics_with_any_context"] != len(official_records):
        raise ValueError(
            "Some selected topics lack indexed same-conversation background: "
            + json.dumps(topic_context_summary, sort_keys=True)
        )
    try:
        (
            synthetic_summary,
            leakage,
            synthetic_relevant_evidence,
        ) = write_dataset(
            official_records,
            relevant_evidence_targets,
            relevant_turn_counts,
            zero_evidence_indices,
            args.output,
            args.audit_output,
            args.seed,
            background,
            topic_phrases,
        )
    finally:
        background.close()
    comparison = distribution_comparison(official_summary, synthetic_summary)
    relevant_comparison = relevant_evidence_comparison(
        official_relevant_evidence,
        synthetic_relevant_evidence,
    )
    if leakage["exact_official_question_overlap_count"]:
        raise ValueError("Generated questions overlap official questions exactly")
    if leakage["exact_official_long_answer_overlap_count"]:
        raise ValueError("Generated long answers overlap official answers exactly")
    if leakage["exact_official_turn_overlap_count_min_5_words"]:
        raise ValueError("Generated turns overlap official turns exactly")
    if not comparison["passed"]:
        raise ValueError(
            "Generated dataset failed distribution checks: "
            + json.dumps(comparison["checks"], sort_keys=True)
        )
    if not relevant_comparison["passed"]:
        raise ValueError(
            "Generated dataset failed relevant-evidence density checks: "
            + json.dumps(relevant_comparison["checks"], sort_keys=True)
        )
    manifest = build_manifest(
        args.official_data,
        args.official_oracle,
        args.output,
        args.audit_output,
        args.seed,
        official_summary,
        synthetic_summary,
        comparison,
        official_relevant_evidence,
        synthetic_relevant_evidence,
        relevant_comparison,
        leakage,
        args.background_manifest,
        topic_context_summary,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote dataset: {args.output}")
    print(f"Wrote audit: {args.audit_output}")
    print(f"Wrote manifest: {args.manifest_output}")
    print(json.dumps({
        "records": synthetic_summary["records"],
        "question_types": synthetic_summary["question_type_counts"],
        "abstentions": synthetic_summary["abstention_count"],
        "source_words_mean": synthetic_summary["source_words"]["mean"],
        "turns_mean": synthetic_summary["turn_count"]["mean"],
        "gold_session_words_mean": synthetic_summary["gold_session_words"]["mean"],
        "distribution_checks": comparison["checks"],
        "relevant_evidence_checks": relevant_comparison["checks"],
        "relevant_evidence_overall": relevant_comparison["overall"],
        "assistant_relevant_evidence": relevant_comparison[
            "by_question_type"
        ]["single-session-assistant"],
        "leakage_checks": leakage,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
