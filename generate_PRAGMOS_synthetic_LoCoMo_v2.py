#!/usr/bin/env python3
"""Generate a high-fidelity, independent LoCoMo development set V2.

V2 mirrors anonymous structural and difficulty profiles from the pinned public
LoCoMo release. It never emits official questions, answers, turns, speakers,
or dates. Independent UltraChat prose supplies the conversational background;
all memory facts and QA items are generated locally and deterministically.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import generate_PRAGMOS_synthetic_LoCoMo as v1
from analyze_PRAGMOS_synthetic_LoCoMo_v2 import (
    analyze_dataset,
    compare_reports,
    normalized,
    sha256_path,
    words,
)
from generate_PRAGMOS_synthetic_LongMemEval_v2 import BackgroundProvider


DEFAULT_SEED = 20261027
OFFICIAL_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
SESSION_RE = re.compile(r"session_(\d+)$")
STOPWORDS = {
    "a", "about", "after", "and", "are", "as", "at", "be", "before",
    "by", "did", "do", "does", "during", "for", "from", "had", "has",
    "have", "how", "i", "in", "is", "it", "my", "of", "on", "or",
    "the", "their", "them", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "would",
}

COLORS = (
    "amber", "aqua", "cobalt", "coral", "emerald", "indigo", "ivory",
    "lavender", "marigold", "navy", "ochre", "plum", "rose", "silver",
    "teal", "violet",
)
OBJECTS = (
    "field notebook", "ceramic mug", "camping lantern", "recipe binder",
    "travel tripod", "fountain pen", "woven scarf", "camera pouch",
    "garden toolkit", "recording microphone", "chess set", "rain jacket",
)
ACTIVITIES = (
    "birdwatching", "bread baking", "calligraphy", "ceramics", "kayaking",
    "oral-history recording", "salsa dancing", "trail running",
    "urban gardening", "watercolor painting", "wood carving", "yoga",
)
VENUES = (
    "Beacon Arts Annex", "Cedar Reading Room", "Harbor Learning Studio",
    "Juniper Community Hall", "Maple Workshop Loft", "Northwind Gallery",
    "Orchid Activity Center", "Pioneer Theatre Lab", "Riverside Pavilion",
    "Silver Lake Classroom", "Willow Market Hall", "Aster Music Room",
)
PEOPLE = (
    "Anika", "Basil", "Clara", "Dev", "Esme", "Farah", "Gideon",
    "Hugo", "Imani", "Jules", "Keira", "Milan", "Nadia", "Owen",
    "Pia", "Ravi", "Selene", "Tomas", "Uma", "Yara",
)
TRAITS = (
    "patient", "resourceful", "community-minded", "curious", "dependable",
    "empathetic", "reflective", "persistent", "practical", "adventurous",
)
CAREERS = (
    "community arts coordinator", "conservation educator", "museum archivist",
    "occupational therapist", "public-interest researcher", "school counselor",
    "urban garden designer", "wildlife rehabilitation assistant",
    "oral historian", "accessible-technology trainer", "nutrition educator",
    "library program manager",
)
CITY_COUNTRY = (
    ("Braga", "Portugal"), ("Busan", "South Korea"), ("Cusco", "Peru"),
    ("Ghent", "Belgium"), ("Kigali", "Rwanda"), ("Kyoto", "Japan"),
    ("Ljubljana", "Slovenia"), ("Marrakesh", "Morocco"),
    ("Nairobi", "Kenya"), ("Oslo", "Norway"), ("Quito", "Ecuador"),
    ("Tallinn", "Estonia"),
)
CITY_REGION = (
    ("Austin", "Texas"), ("Boulder", "Colorado"), ("Eugene", "Oregon"),
    ("Madison", "Wisconsin"), ("Portland", "Maine"),
    ("Richmond", "Virginia"), ("Savannah", "Georgia"),
    ("Seattle", "Washington"), ("Tucson", "Arizona"),
    ("Burlington", "Vermont"), ("Fresno", "California"),
    ("Boise", "Idaho"),
)


@dataclass(frozen=True)
class QAProfile:
    question_words: int
    answer_words: int
    evidence_count: int
    evidence_sessions: int
    caption_evidence: bool
    caption_only_answer: bool
    answer_exact_in_evidence: bool


def word_count(value: object) -> int:
    return len(WORD_RE.findall(str(value or "")))


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(value.title(), "%I:%M %p On %d %B, %Y")


def format_date(value: datetime) -> str:
    return f"{value.day} {value:%B %Y}"


def add_month(value: datetime) -> datetime:
    year = value.year + int(value.month == 12)
    month = 1 if value.month == 12 else value.month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def previous_weekday(value: datetime, weekday: int) -> datetime:
    distance = (value.weekday() - weekday) % 7 or 7
    return value - timedelta(days=distance)


def official_profiles(dataset: list[dict]) -> list[dict[int, list[QAProfile]]]:
    profiles = []
    for sample in dataset:
        conversation = sample["conversation"]
        dialog_map = {
            str(turn.get("dia_id")): turn
            for key, session in conversation.items()
            if SESSION_RE.fullmatch(key) and isinstance(session, list)
            for turn in session
        }
        by_category: dict[int, list[QAProfile]] = defaultdict(list)
        for qa in sample.get("qa") or []:
            evidence_ids = [str(item) for item in qa.get("evidence") or []]
            evidence_rows = [
                dialog_map[item] for item in evidence_ids if item in dialog_map
            ]
            evidence_text = " ".join(
                f"{turn.get('text') or ''} {turn.get('blip_caption') or ''}"
                for turn in evidence_rows
            )
            plain_text = " ".join(
                str(turn.get("text") or "") for turn in evidence_rows
            )
            answer = normalized(qa.get("answer"))
            by_category[int(qa["category"])].append(
                QAProfile(
                    question_words=word_count(qa.get("question")),
                    answer_words=word_count(qa.get("answer")),
                    evidence_count=len(evidence_ids),
                    evidence_sessions=len(
                        {item.split(":", 1)[0] for item in evidence_ids}
                    ),
                    caption_evidence=any(
                        turn.get("blip_caption") for turn in evidence_rows
                    ),
                    caption_only_answer=bool(
                        answer
                        and answer in normalized(evidence_text)
                        and answer not in normalized(plain_text)
                    ),
                    answer_exact_in_evidence=bool(
                        answer and answer in normalized(evidence_text)
                    ),
                )
            )
        profiles.append(dict(by_category))
    return profiles


def official_turn_word_targets(dataset: list[dict]) -> list[list[int]]:
    rows = []
    for sample in dataset:
        values = []
        for key, session in sorted(
            (
                (key, value)
                for key, value in sample["conversation"].items()
                if SESSION_RE.fullmatch(key) and isinstance(value, list)
            ),
            key=lambda item: int(SESSION_RE.fullmatch(item[0]).group(1)),
        ):
            values.extend(word_count(turn.get("text")) for turn in session)
        rows.append(values)
    return rows


class SampleState:
    def __init__(self, sample: dict, turn_word_targets: list[int], seed: int):
        self.sample = sample
        self.conversation = sample["conversation"]
        self.speakers = (
            self.conversation["speaker_a"],
            self.conversation["speaker_b"],
        )
        self.sessions: dict[int, list[dict]] = {}
        self.session_dates: dict[int, datetime] = {}
        self.turns: dict[str, tuple[int, dict]] = {}
        self.usage: Counter[str] = Counter()
        self.decoy_usage: Counter[str] = Counter()
        self.noncaption_evidence: set[str] = set()
        self.rng = random.Random(seed)
        target_iterator = iter(turn_word_targets)
        for key, value in sorted(
            self.conversation.items(),
            key=lambda item: (
                int(SESSION_RE.fullmatch(item[0]).group(1))
                if SESSION_RE.fullmatch(item[0])
                else 10_000
            ),
        ):
            match = SESSION_RE.fullmatch(key)
            if match and isinstance(value, list):
                session_number = int(match.group(1))
                self.sessions[session_number] = value
                self.session_dates[session_number] = parse_datetime(
                    self.conversation[f"session_{session_number}_date_time"]
                )
                for turn in value:
                    turn["_target_words"] = next(target_iterator)
                    turn.pop("blip_caption", None)
                    turn.pop("img_url", None)
                    turn.pop("query", None)
                    turn["text"] = ""
                    turn["_facts"] = []
                    turn["_topics"] = []
                    self.turns[str(turn["dia_id"])] = (session_number, turn)

    def select_evidence(
        self,
        count: int,
        session_count: int,
        actors: list[str],
        salt: object,
        *,
        caption_required: bool = False,
    ) -> list[str]:
        if count <= 0:
            return []
        session_count = max(1, min(session_count, count, len(self.sessions)))
        session_numbers = sorted(self.sessions)
        start = stable_seed(salt, self.speakers) % len(session_numbers)
        chosen_sessions = [
            session_numbers[(start + index * 5) % len(session_numbers)]
            for index in range(session_count)
        ]
        chosen_sessions = list(dict.fromkeys(chosen_sessions))
        cursor = 0
        while len(chosen_sessions) < session_count:
            candidate = session_numbers[(start + cursor) % len(session_numbers)]
            if candidate not in chosen_sessions:
                chosen_sessions.append(candidate)
            cursor += 1
        selected: list[str] = []
        for index in range(count):
            session_number = chosen_sessions[index % len(chosen_sessions)]
            actor = actors[index % len(actors)]
            candidates = [
                turn
                for turn in self.sessions[session_number]
                if turn.get("speaker") == actor
                and str(turn["dia_id"]) not in selected
            ]
            if not candidates:
                candidates = [
                    turn
                    for turn in self.sessions[session_number]
                    if str(turn["dia_id"]) not in selected
                ]
            if caption_required:
                compatible = [
                    turn
                    for turn in candidates
                    if self.usage[str(turn["dia_id"])] == 0
                    and not turn.get("blip_caption")
                    and str(turn["dia_id"]) not in self.noncaption_evidence
                ]
                if not compatible and count == 1:
                    compatible = [
                        turn
                        for _, turn in self.turns.values()
                        if turn.get("speaker") == actor
                        and self.usage[str(turn["dia_id"])] == 0
                        and not turn.get("blip_caption")
                        and str(turn["dia_id"])
                        not in self.noncaption_evidence
                    ]
            else:
                compatible = [
                    turn for turn in candidates if not turn.get("blip_caption")
                ]
            if compatible:
                candidates = compatible
            turn = min(
                candidates,
                key=lambda value: (
                    bool(value.get("blip_caption")) != caption_required,
                    (
                        caption_required
                        and str(value["dia_id"]) in self.noncaption_evidence
                    ),
                    self.usage[str(value["dia_id"])],
                    -int(value["_target_words"]),
                    int(str(value["dia_id"]).split(":", 1)[1]),
                ),
            )
            dialog_id = str(turn["dia_id"])
            selected.append(dialog_id)
            self.usage[dialog_id] += 1
            if not caption_required:
                self.noncaption_evidence.add(dialog_id)
        return selected

    def add_fact(
        self,
        dialog_id: str,
        text: str,
        topic: str,
        *,
        caption: str | None = None,
    ) -> None:
        _, turn = self.turns[dialog_id]
        if text:
            turn["_facts"].append(text.strip())
        turn["_topics"].append(topic)
        if caption:
            turn["blip_caption"] = caption.strip()
            turn["synthetic_media"] = True

    def add_decoys(
        self,
        question: str,
        answer: object,
        topic: str,
        evidence_ids: list[str],
        count: int = 2,
    ) -> None:
        answer_tokens = set(words(answer))
        anchors = [
            token
            for token in words(question)
            if token not in STOPWORDS
            and token not in answer_tokens
            and len(token) >= 4
        ][:2]
        if not anchors:
            return
        blocked = set(evidence_ids)
        candidates = [
            (dialog_id, turn)
            for dialog_id, (_, turn) in self.turns.items()
            if dialog_id not in blocked
        ]
        for _ in range(min(count, len(candidates))):
            dialog_id, turn = min(
                candidates,
                key=lambda value: (
                    bool(value[1].get("blip_caption")),
                    self.decoy_usage[value[0]],
                    self.usage[value[0]],
                    -int(value[1]["_target_words"]),
                    value[0],
                ),
            )
            candidates.remove((dialog_id, turn))
            self.decoy_usage[dialog_id] += 1
            phrase = " and ".join(anchors)
            openings = (
                "Separately", "Elsewhere", "Incidentally", "Briefly",
                "Earlier", "Later", "Meanwhile", "Tangentially",
                "Additionally", "Casually", "Secondarily", "Momentarily",
                "During planning", "In follow-up", "As context", "Off-topic",
            )
            links = (
                "surfaced around", "came up beside", "was mentioned with",
                "was linked to", "appeared near", "was compared with",
                "was raised during", "was noted alongside", "accompanied",
                "was considered in", "was recalled with", "was listed under",
                "was flagged beside", "was associated with", "briefly entered",
                "was questioned during",
            )
            endings = (
                "tentatively", "unresolved", "without detail", "in passing",
                "without confirmation", "only briefly", "inconclusively",
                "as background", "without specifics", "without a decision",
                "as an aside", "without an outcome", "then faded",
                "without commitment", "as context", "without resolution",
            )
            decoy_seed = stable_seed(dialog_id, question)
            opening = openings[decoy_seed % len(openings)]
            link = links[(decoy_seed // len(openings)) % len(links)]
            ending = endings[
                (decoy_seed // (len(openings) * len(links))) % len(endings)
            ]
            turn["_facts"].append(
                f"{opening}, {phrase} {link} {topic} {ending}."
            )
            turn["_topics"].append(topic)

    def date_for(self, dialog_id: str) -> datetime:
        return self.session_dates[self.turns[dialog_id][0]]


def fit_question(text: str, target_words: int, rng: random.Random) -> str:
    text = text.rstrip("?").strip()
    compact_replacements = (
        (r"\bwhat complete activity list did\b", "which activities did"),
        (r"\bwhat activity list did\b", "which activities did"),
        (r"\bduring the stages of\b", "during"),
        (r"\bhow many unique workshop entries belong to\b", "how many workshops did"),
        (r"\bwhat activity do\b", "what do"),
        (r"\bwhat is the contents list for\b", "what is in"),
        (r"\bwhat is the complete contents list for\b", "what is in"),
        (r"\bwhat is the combined reading list from\b", "what did"),
        (r"\bwhat complete set of suggested items appears in\b", "what was suggested in"),
        (r"\bon what date did\b", "when did"),
        (r"\bduring which season did\b", "when did"),
        (r"\bacross the separate\b", "across"),
        (r"\bacross the different\b", "across"),
        (r"\bacross the conversations\b", "across chats"),
        (r"\bacross the sessions\b", "across sessions"),
        (r"\bover the course of\b", "during"),
        (r"\bin connection with\b", "in"),
        (r"\bwhen the separate updates are combined\b", "across updates"),
        (r"\bcombining the separate messages\b", "across messages"),
        (r"\baccording to the conversation\b", "in the conversation"),
        (r"\bfrom the relevant messages\b", "in the messages"),
        (r"\bwhat is the number of\b", "how many"),
        (r"\bwhat complete set of\b", "which"),
        (r"\bwhat full set of\b", "which"),
    )
    for pattern, replacement in compact_replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    removable = (
        "separate", "different", "relevant", "earlier",
        "main", "broad", "practical",
    )
    for token in removable:
        if word_count(text) <= target_words + 1:
            break
        text = re.sub(rf"\b{token}\b\s*", "", text, count=1, flags=re.IGNORECASE)
    if re.match(r"^Name\b", text, flags=re.IGNORECASE):
        text = re.sub(
            r"^Name\b",
            rng.choice(("Name", "Identify", "State", "Recall")),
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    synonym_sets = {
        "activity": ("activity", "practice", "pursuit", "pastime"),
        "item": ("item", "object", "article", "belonging"),
        "image": ("image", "photo", "picture", "snapshot"),
        "picture": ("picture", "image", "photo", "snapshot"),
        "meeting": ("meeting", "session", "gathering", "appointment"),
        "location": ("location", "place", "site", "venue"),
        "goal": ("goal", "aim", "objective", "target"),
        "advice": ("advice", "guidance", "recommendation", "suggestion"),
        "group": ("group", "circle", "organization", "association"),
        "exact": ("exact", "precise", "specific", "definite"),
        "result": ("result", "outcome", "finding", "score"),
        "event": ("event", "gathering", "occasion", "session"),
        "problem": ("problem", "issue", "obstacle", "difficulty"),
        "reason": ("reason", "cause", "explanation", "basis"),
        "future": ("future", "later", "upcoming", "subsequent"),
        "course": ("course", "class", "program", "training"),
        "electronic": ("electronic", "email", "digital", "online"),
        "short": ("short", "brief", "concise", "compact"),
        "line": ("line", "sentence", "phrase", "quotation"),
        "table": ("table", "grid", "sheet", "matrix"),
        "row": ("row", "line", "entry", "record"),
        "website": ("website", "webpage", "site", "portal"),
        "specific": ("specific", "exact", "precise", "particular"),
        "obstacle": ("obstacle", "problem", "issue", "barrier"),
        "confirmed": ("confirmed", "set", "finalized", "recorded"),
        "date": ("date", "day", "calendar date", "calendar day"),
        "hour": ("hour", "time", "clock time", "time slot"),
        "companion": ("companion", "partner", "company", "fellow traveler"),
        "attendance": ("attendance", "turnout", "participation", "headcount"),
        "program": ("program", "course", "training", "class"),
        "model": ("model", "version", "device", "variant"),
        "final": ("final", "last", "concluding", "ending"),
        "career": ("career", "profession", "occupation", "career path"),
        "genre": ("genre", "style", "category", "literary type"),
        "tool": ("tool", "instrument", "equipment", "implement"),
        "traits": ("traits", "qualities", "characteristics", "attributes"),
        "holiday": ("holiday", "celebration", "occasion", "festival"),
    }
    for source, choices in synonym_sets.items():
        if re.search(rf"\b{source}\b", text, flags=re.IGNORECASE):
            replacement = rng.choice(choices)
            text = re.sub(
                rf"\b{source}\b",
                replacement,
                text,
                flags=re.IGNORECASE,
            )
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text[0].upper() + text[1:] + "?"


def choose(values: tuple[str, ...], ordinal: int, step: int = 0) -> str:
    return values[(ordinal * 7 + step * 5) % len(values)]


def keyed_choice(values: tuple[str, ...], *keys: object) -> str:
    return values[stable_seed(*keys) % len(values)]


def topic_tag(topic: str) -> str:
    tokens = words(topic)[:2]
    return "-".join(tokens).title() or "Project"


def with_indefinite_article(phrase: str) -> str:
    article = "an" if phrase[:1].lower() in "aeiou" else "a"
    return f"{article} {phrase}"


def replace_topic_reference(text: str, topic: str, *keys: object) -> str:
    references = (
        "the effort", "that plan", "the activity", "that work",
        "the initiative", "that project", "the undertaking", "the program",
    )
    return text.replace(topic, keyed_choice(references, topic, *keys))


def answer_with_reason(answer: str, reason: str, target_words: int) -> str:
    if target_words <= word_count(answer) + 2:
        return answer
    candidate = f"{answer}, because {reason}"
    if word_count(candidate) <= max(target_words + 3, word_count(answer) + 5):
        return candidate
    return answer


def set_qa(
    qa: dict,
    *,
    question: str,
    answer: object,
    evidence: list[str],
    family: str,
    topic: str,
) -> None:
    qa["question"] = question
    qa["answer"] = answer
    qa["evidence"] = evidence
    qa["synthetic_family"] = family
    qa["_topic"] = topic


def generate_multi_hop(
    qa: dict,
    profile: QAProfile,
    state: SampleState,
    topic: str,
    ordinal: int,
    rng: random.Random,
) -> None:
    actor = state.speakers[ordinal % 2]
    other = state.speakers[1 - ordinal % 2]
    family = ordinal % 12
    actors = [actor]
    if family == 3:
        actors = [actor, other]
    evidence = state.select_evidence(
        profile.evidence_count,
        profile.evidence_sessions,
        actors,
        ("multi", ordinal),
        caption_required=profile.caption_evidence,
    )
    count = max(1, len(evidence))
    value_count = min(count, max(1, (profile.answer_words + 1) // 3))
    tag = topic_tag(topic)
    statements: list[str] = []

    if family == 0:
        values = [
            f"{keyed_choice(COLORS, topic, ordinal, i, 'activity-color')} "
            f"{keyed_choice(ACTIVITIES, topic, ordinal, i, 'activity')}"
            for i in range(value_count)
        ]
        statements = [
            f"I added {values[i % value_count]} to my regular activities."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"Which {topic} activities did {actor} add",
            f"What new {topic} activities did {actor} try",
            f"What did {actor} take up during {topic}",
            f"List {actor}'s activities from {topic}",
        ])
        family_name = "activity-history-union"
    elif family == 1:
        values = [
            f"{keyed_choice(VENUES, topic, ordinal, i, 'venue')}"
            for i in range(value_count)
        ]
        statements = [
            f"One place I visited was {values[i % value_count]}."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"Which {topic} places did {actor} visit",
            f"What locations did {actor} mention for {topic}",
            f"Where did {actor} go during {topic}",
            f"List {actor}'s destinations from {topic}",
        ])
        family_name = "place-history-union"
    elif family == 2:
        values = [f"{tag} workshop {i + 1}" for i in range(value_count)]
        statements = [
            f"I completed {values[i % value_count]}." for i in range(count)
        ]
        workshop_count = len(set(values))
        workshop_noun = "workshop" if workshop_count == 1 else "workshops"
        answer = f"{workshop_count} distinct {tag} {workshop_noun}"
        question = rng.choice([
            f"How many {topic} workshops did {actor} complete",
            f"What is {actor}'s distinct workshop count for {topic}",
            f"Counting once, how many {topic} workshops did {actor} finish",
            f"What total of unique {topic} workshops belongs to {actor}",
        ])
        family_name = "cross-session-count-distinct"
    elif family == 3:
        shared = (
            f"{keyed_choice(COLORS, topic, ordinal, 'shared-color')} "
            f"{keyed_choice(ACTIVITIES, topic, ordinal, 'shared-activity')}"
        )
        statements = [
            f"For {topic}, one activity I especially enjoy is {shared}."
            for _ in evidence
        ]
        answer = shared
        question = rng.choice([
            f"What did {actor} and {other} both enjoy during {topic}",
            f"Which {topic} interest links {actor} and {other}",
            f"What common {topic} activity did {actor} and {other} mention",
            f"Where did {actor}'s and {other}'s {topic} interests overlap",
        ])
        family_name = "speaker-set-intersection"
    elif family == 4:
        values = [
            f"{keyed_choice(COLORS, topic, ordinal, i, 'archive-color')} "
            f"{keyed_choice(OBJECTS, topic, ordinal, i, 'archive-object')}"
            for i in range(value_count)
        ]
        statements = [
            f"My archive now contains {values[i % value_count]}."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"What is in {actor}'s {topic} archive",
            f"Which {topic} items did {actor} collect",
            f"List the contents of {actor}'s {topic} archive",
            f"What did {actor} place in the {topic} collection",
        ])
        family_name = "collection-union"
    elif family == 5:
        values = [
            f"complete {2 + ((ordinal + i) % 8)} "
            f"{keyed_choice(ACTIVITIES, topic, ordinal, i, 'goal')} sessions"
            for i in range(value_count)
        ]
        statements = [
            f"One goal I set was to {values[i % value_count]}."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"What goals did {actor} set during {topic}",
            f"Which objectives formed {actor}'s {topic} plan",
            f"What did {actor} intend to accomplish in {topic}",
            f"List the {topic} goals described by {actor}",
        ])
        family_name = "goal-synthesis"
    elif family == 6:
        values = [
            f"{keyed_choice(COLORS, topic, ordinal, i, 'recommend-color')} "
            f"{keyed_choice(OBJECTS, topic, ordinal, i, 'recommend-object')}"
            for i in range(value_count)
        ]
        statements = [
            f"{keyed_choice(PEOPLE, topic, ordinal, i, 'recommender')} "
            f"recommended {values[i % value_count]} to me."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"What was recommended to {actor} during {topic}",
            f"Which {topic} recommendations did {actor} receive",
            f"What suggested items appear in {actor}'s {topic} record",
            f"What was {actor} advised to get for {topic}",
        ])
        family_name = "recommendation-union"
    elif family == 7:
        values = [
            f"{keyed_choice(ACTIVITIES, topic, ordinal, i, 'family-activity')} "
            f"with {keyed_choice(PEOPLE, topic, ordinal, i, 'family-person')}"
            for i in range(value_count)
        ]
        statements = [
            f"My family and I spent time {values[i % value_count]}."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"What did {actor} do with family during {topic}",
            f"How did {actor}'s family spend time in {topic}",
            f"Which family activities appear in {actor}'s {topic} messages",
            f"List {actor}'s shared family activities from {topic}",
        ])
        family_name = "family-activity-synthesis"
    elif family == 8:
        values = [
            f"The {keyed_choice(COLORS, topic, ordinal, i, 'title-color').title()} "
            f"{keyed_choice(OBJECTS, topic, ordinal, i, 'title-object').title()}"
            for i in range(value_count)
        ]
        statements = [
            f"I finished reading a memoir called {values[i % value_count]}."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"What did {actor} finish reading during {topic}",
            f"Which books appear in {actor}'s {topic} updates",
            f"List the titles from {actor}'s {topic} conversations",
            f"What books did {actor} read for {topic}",
        ])
        family_name = "reading-history-union"
    elif family == 9:
        values = [
            keyed_choice(PEOPLE, topic, ordinal, i, "supporter")
            for i in range(value_count)
        ]
        statements = [
            f"When the work became difficult, {values[i % value_count]} supported me."
            for i in range(count)
        ]
        answer = ", ".join(values)
        question = rng.choice([
            f"Who supported {actor} during {topic}",
            f"Who formed {actor}'s {topic} support network",
            f"Whom did {actor} call helpful in {topic}",
            f"Who could {actor} rely on during {topic}",
        ])
        family_name = "support-network-synthesis"
    elif family == 10:
        values = [
            f"{keyed_choice(COLORS, topic, ordinal, i, 'route')} route {i + 1}"
            for i in range(count)
        ]
        statements = [
            (
                f"At that point in {topic}, my preferred route was {value}."
                if i == 0
                else f"I later changed my {topic} route to {value}, replacing the earlier choice."
            )
            for i, value in enumerate(values)
        ]
        answer = values[-1]
        question = rng.choice([
            f"Which route is current for {actor}'s {topic}",
            f"What final route replaced {actor}'s {topic} choices",
            f"Which route ended {actor}'s {topic} history",
            f"What was {actor}'s latest {topic} route",
        ])
        family_name = "cross-session-state-update"
    else:
        values = [
            f"the {keyed_choice(COLORS, topic, ordinal, i, 'event')} community event"
            for i in range(count)
        ]
        statements = [f"I attended {value}." for value in values]
        answer = f"{count} {tag} events"
        question = rng.choice([
            f"How many {topic} events did {actor} attend",
            f"What was {actor}'s attendance count for {topic}",
            f"How often did {actor} attend {with_indefinite_article(f'{topic} event')}",
            f"What is the total of {actor}'s completed {topic} events",
        ])
        family_name = "cross-session-frequency-count"

    statements = [
        replace_topic_reference(statement, topic, "multi", ordinal, index)
        for index, statement in enumerate(statements)
    ]
    for index, dialog_id in enumerate(evidence):
        state.add_fact(dialog_id, statements[index], topic)
    if profile.caption_evidence and evidence:
        state.add_fact(
            evidence[0],
            "",
            topic,
            caption="a contextual photograph connected with the same project",
        )
    if word_count(answer) < profile.answer_words:
        answer = f"Overall: {answer}"
    question = fit_question(question, profile.question_words, rng)
    set_qa(
        qa,
        question=question,
        answer=answer,
        evidence=evidence,
        family=family_name,
        topic=topic,
    )


def generate_temporal(
    qa: dict,
    profile: QAProfile,
    state: SampleState,
    topic: str,
    ordinal: int,
    rng: random.Random,
) -> None:
    actor = state.speakers[ordinal % 2]
    evidence = state.select_evidence(
        profile.evidence_count,
        profile.evidence_sessions,
        [actor],
        ("temporal", ordinal),
        caption_required=profile.caption_evidence,
    )
    anchor_id = evidence[0]
    source_date = state.date_for(anchor_id)
    family = ordinal % 12
    event = rng.choice(
        [
            "community presentation", "archive review", "training session",
            "gallery visit", "planning meeting", "volunteer shift",
            "workshop showcase", "route inspection", "book-club gathering",
            "fundraising dinner", "field interview", "design rehearsal",
        ]
    )
    statements: list[str] = []
    if family == 0:
        answer = format_date(source_date - timedelta(days=1))
        statements.append(f"Yesterday I completed the {event}.")
        question = f"When did {actor} complete the {topic} {event}"
        family_name = "relative-yesterday"
    elif family == 1:
        answer = f"The week before {format_date(source_date)}"
        statements.append(f"Last week I attended the {event}.")
        question = f"When did {actor} attend the {topic} {event}"
        family_name = "relative-last-week"
    elif family == 2:
        answer = add_month(source_date).strftime("%B %Y")
        statements.append(f"Next month I plan to hold the {event}.")
        question = f"When is {actor}'s {topic} {event} planned"
        family_name = "relative-next-month"
    elif family == 3:
        days = 2 + ordinal % 8
        answer = format_date(source_date - timedelta(days=days))
        statements.append(f"{days} days ago I finished the {event}.")
        question = f"When did {actor} finish the {topic} {event}"
        family_name = "relative-days-ago"
    elif family == 4:
        years = 2 + ordinal % 7
        answer = f"a {years}-year {topic_tag(topic)} history"
        statements.append(f"I have maintained the archive for {years} years.")
        question = f"How long is {actor}'s {topic} archive history"
        family_name = "explicit-duration"
    elif family == 5:
        answer_date = previous_weekday(source_date, ordinal % 5)
        weekday = answer_date.strftime("%A")
        answer = format_date(answer_date)
        statements.append(f"Last {weekday} I presented the proposal.")
        question = f"When did {actor} present the {topic} proposal"
        family_name = "relative-weekday"
    elif family == 6:
        year = 2018 + ordinal % 8
        answer = str(year)
        statements.append(f"I first drafted the plan {source_date.year - year} years ago.")
        question = f"Which year did {actor} draft the first {topic} plan"
        family_name = "explicit-year"
    elif family == 7:
        answer_date = source_date - timedelta(days=8 + ordinal % 15)
        answer = format_date(answer_date)
        statements.append(f"The {event} took place on {answer}.")
        question = f"What date was {actor}'s {topic} {event}"
        family_name = "explicit-date"
    elif family == 8:
        answer_date = source_date + timedelta(days=max(1, len(evidence)))
        answer = format_date(answer_date)
        statements.append("Today I attended the kickoff.")
        for index in range(1, len(evidence)):
            statements.append(
                "The next checkpoint happened one day after the previous stage."
            )
        question = f"When was {actor}'s final {topic} checkpoint"
        family_name = "cross-session-relative-date-chain"
    elif family == 9:
        days = 3 + ordinal % 12
        answer = f"a {days}-day {topic_tag(topic)} review"
        start_statement = (
            "I began the review on "
            f"{format_date(source_date - timedelta(days=days))}."
        )
        end_statement = "I completed the review today."
        if len(evidence) == 1:
            statements.append(f"{start_statement} {end_statement}")
        else:
            statements.extend((start_statement, end_statement))
        question = f"How long was {actor}'s {topic} review"
        family_name = "cross-session-duration"
    elif family == 10:
        first = source_date - timedelta(days=14)
        second = source_date - timedelta(days=4)
        answer = f"the planning meeting, then the {event}"
        first_statement = (
            f"The planning meeting happened on {format_date(first)}."
        )
        second_statement = (
            f"The {event} happened on {format_date(second)}."
        )
        if len(evidence) == 1:
            statements.append(f"{first_statement} {second_statement}")
        else:
            statements.extend((first_statement, second_statement))
        question = f"Which came first in {actor}'s {topic}: planning or the {event}"
        family_name = "temporal-event-order"
    else:
        season = ("spring", "summer", "autumn", "winter")[ordinal % 4]
        answer = f"{season} {source_date.year}"
        month_name = ("March", "June", "September", "December")[ordinal % 4]
        statements.append(f"I scheduled the {event} in {month_name} {source_date.year}.")
        question = f"Which season was {actor}'s {topic} {event} scheduled"
        family_name = "season-resolution"

    while len(statements) < len(evidence):
        statements.append(
            "A later message confirmed the same timing without changing it."
        )
    for index, dialog_id in enumerate(evidence):
        state.add_fact(dialog_id, statements[index], topic)
    if profile.caption_evidence:
        state.add_fact(
            evidence[0], "", topic,
            caption=f"a dated photograph associated with the {event} for {topic}",
        )
    question = fit_question(question, profile.question_words, rng)
    set_qa(
        qa,
        question=question,
        answer=answer,
        evidence=evidence,
        family=family_name,
        topic=topic,
    )


def generate_open_domain(
    qa: dict,
    profile: QAProfile,
    state: SampleState,
    topic: str,
    ordinal: int,
    rng: random.Random,
) -> None:
    actor = state.speakers[ordinal % 2]
    evidence = state.select_evidence(
        profile.evidence_count,
        profile.evidence_sessions,
        [actor],
        ("open", ordinal),
        caption_required=profile.caption_evidence,
    )
    labeled_evidence = list(evidence)
    if not evidence:
        evidence = state.select_evidence(1, 1, [actor], ("open-hidden", ordinal))
    family = ordinal % 16
    premises: list[str]
    tag = topic_tag(topic)

    if family == 0:
        city, country = keyed_choice(CITY_COUNTRY, topic, ordinal, "country")
        premises = [f"During {topic}, I spent several days exploring {city} and its older neighborhoods."]
        answer = f"{country}, inferred from {city} in {tag}"
        question = f"Which country did {actor} visit during {topic}"
        family_name = "geographic-country-inference"
    elif family == 1:
        city, region = keyed_choice(CITY_REGION, topic, ordinal, "region")
        premises = [f"For {topic}, I stayed with friends in {city} and traveled around the surrounding area."]
        answer = f"{region}, inferred from {city} in {tag}"
        question = f"Which US state did {actor} visit for {topic}"
        family_name = "geographic-state-inference"
    elif family == 2:
        career = keyed_choice(CAREERS, topic, ordinal, "career")
        interest = keyed_choice(ACTIVITIES, topic, ordinal, "interest")
        premises = [
            f"During {topic}, I enjoyed teaching beginners about {interest} and organizing accessible lessons.",
            "I said that helping people learn mattered more to me than selling products.",
        ]
        answer = f"{career} fits the {tag} interests"
        question = f"What career might suit {actor} based on the {topic} conversations"
        family_name = "career-inference"
    elif family == 3:
        object_name = with_indefinite_article(
            f"{keyed_choice(COLORS, topic, ordinal, 'gift-color')} "
            f"{keyed_choice(OBJECTS, topic, ordinal, 'gift-object')}"
        )
        habit = keyed_choice(ACTIVITIES, topic, ordinal, "gift-habit")
        premises = [f"While discussing {topic}, I said I had begun {habit} every weekend and wanted to track my progress."]
        answer = f"{object_name} for {habit} in {tag}"
        question = f"What practical gift would support {actor}'s new habit from {topic}"
        family_name = "commonsense-gift-inference"
    elif family == 4:
        genres = (
            "historical mystery", "science fiction", "travel memoir",
            "nature writing", "character-driven fantasy", "social history",
            "narrative journalism", "literary adventure",
        )
        genre = keyed_choice(genres, topic, ordinal, "genre")
        clues = (
            "archival puzzles and old family secrets",
            "spacecraft, distant planets, and future settlements",
            "first-person journeys through unfamiliar cities",
            "wildlife, landscapes, and conservation work",
            "imagined worlds centered on close friendships",
            "ordinary people living through major social change",
            "deeply reported accounts of real events",
            "long journeys with difficult practical choices",
        )
        clue = clues[genres.index(genre)]
        premises = [f"Throughout {topic}, I kept choosing stories about {clue}."]
        answer = f"{genre} fits the {tag} pattern"
        question = f"Which reading genre would {actor} probably enjoy after {topic}"
        family_name = "genre-inference"
    elif family == 5:
        tool_pairs = (
            ("loose cabinet screws and flat-pack shelves", "a screwdriver set"),
            ("fabric panels and a torn canvas bag", "a sewing kit"),
            ("uneven picture frames and shelf brackets", "a spirit level"),
            ("seedlings in dry balcony planters", "a moisture meter"),
            ("interviews recorded in a noisy hall", "a directional microphone"),
            ("maps marked with walking distances", "a measuring wheel"),
        )
        clue, tool = keyed_choice(tool_pairs, topic, ordinal, "tool")
        premises = [f"The practical work for {topic} involved {clue}."]
        answer = f"{tool} best fits {tag}"
        question = f"What tool would {actor} most likely need for {topic}"
        family_name = "tool-inference"
    elif family == 6:
        traits = [
            keyed_choice(TRAITS, topic, ordinal, "trait-1"),
            keyed_choice(TRAITS, topic, ordinal, "trait-2"),
            keyed_choice(TRAITS, topic, ordinal, "trait-3"),
        ]
        premises = [
            f"For {topic}, I kept returning to help after setbacks and listened carefully before changing the plan.",
            "I also made sure quieter volunteers had time to explain their concerns.",
        ]
        answer = f"{traits[0]}, {traits[1]}, and {traits[2]} in {tag}"
        question = f"Which personality traits would someone likely attribute to {actor} after {topic}"
        family_name = "personality-inference"
    elif family == 7:
        outdoor = keyed_choice(ACTIVITIES, topic, ordinal, "outdoor")
        indoor = "a crowded indoor arcade"
        premises = [f"During {topic}, I repeatedly chose quiet outdoor time and especially enjoyed {outdoor}."]
        answer = f"The outdoor option, matching {outdoor} in {tag}"
        question = f"Would {actor} probably choose an outdoor retreat or {indoor} after {topic}"
        family_name = "preference-inference"
    elif family == 8:
        fields = (
            "environmental science", "public health", "museum studies",
            "social work", "urban planning", "education policy",
            "library science", "occupational therapy",
        )
        field = keyed_choice(fields, topic, ordinal, "field")
        premises = [f"In {topic}, I described wanting formal training that combines research with practical community service."]
        answer = f"{with_indefinite_article(field).capitalize()} degree fits {tag}"
        question = f"What degree field might fit {actor}'s goals from {topic}"
        family_name = "education-field-inference"
    elif family == 9:
        pets = (
            ("fur triggers my breathing problems", "a fish or a reptile"),
            ("I react badly to feathers", "a short-haired cat or a fish"),
            ("my building does not allow large animals", "a hamster or a small aquarium"),
            ("I need an animal that requires no outdoor walking", "a pair of gerbils or a fish"),
        )
        clue, compatible = keyed_choice(pets, topic, ordinal, "pet")
        premises = [f"While discussing {topic}, I explained that {clue}."]
        answer = f"{compatible}, given the {tag} constraint"
        question = f"What kind of pet would be more compatible with {actor}'s constraint from {topic}"
        family_name = "constraint-inference"
    elif family == 10:
        premises = [
            f"Throughout {topic}, I volunteered without payment, shared supplies, and returned even when attendance was low.",
            "I said the neighborhood benefit mattered more than personal recognition.",
        ]
        answer = f"Yes; {actor}'s {tag} choices show community commitment"
        question = f"Would {actor} probably support another community project after {topic}"
        family_name = "behavioral-inference"
    elif family == 11:
        premises = [
            f"During {topic}, I relied on buses, repaired belongings instead of replacing them, and compared every fee carefully.",
        ]
        answer = f"{actor} appears budget-conscious in {tag}"
        question = f"What broad financial habit can be inferred about {actor} from {topic}"
        family_name = "financial-behavior-inference"
    elif family == 12:
        relation = keyed_choice(
            ("older cousin", "former instructor", "neighbor", "work mentor"),
            topic,
            ordinal,
            "relation",
        )
        person = keyed_choice(PEOPLE, topic, ordinal, "person")
        premises = [
            f"In the {topic} messages, {person} gave me career advice and checked my applications every week.",
            f"I described {person} as someone I first met through a structured learning program.",
        ]
        answer = f"{person} was likely {actor}'s {relation} in {tag}"
        question = f"What relationship did {person} most likely have to {actor} during {topic}"
        family_name = "relationship-inference"
    elif family == 13:
        premises = [
            f"For {topic}, I arranged child care, renewed the membership, and reserved the same venue for another month.",
        ]
        answer = f"Yes; the {tag} preparations imply continuation"
        question = f"Was {actor} likely planning to continue the activity after {topic}"
        family_name = "future-intention-inference"
    elif family == 14:
        premises = [
            f"Before {topic}, I wanted to leave the field, but encouragement and practical support changed my decision.",
            "I later described the support as the reason I felt able to continue.",
        ]
        answer = f"No; {actor} depended on the {tag} support"
        question = f"Would {actor} still have continued without the support described in {topic}"
        family_name = "counterfactual-inference"
    else:
        holiday_pairs = (
            ("fireworks and a neighborhood parade in early July", "Independence Day"),
            ("costumes and carved pumpkins at the end of October", "Halloween"),
            ("a late-November meal centered on gratitude", "Thanksgiving"),
            ("a December gathering with decorated trees", "Christmas"),
        )
        clue, holiday = keyed_choice(holiday_pairs, topic, ordinal, "holiday")
        premises = [f"The {topic} incident happened around {clue}."]
        answer = f"{holiday}, inferred from the {tag} clues"
        question = f"Around which US holiday did the event in {actor}'s {topic} account occur"
        family_name = "holiday-inference"

    while len(premises) < len(evidence):
        premises.append(
            f"Another {topic} update reinforced the same pattern without stating the conclusion directly."
        )
    premises = [
        replace_topic_reference(premise, topic, "open", ordinal, index)
        for index, premise in enumerate(premises)
    ]
    for index, dialog_id in enumerate(evidence):
        state.add_fact(dialog_id, premises[index], topic)
    if profile.caption_evidence and evidence:
        state.add_fact(
            evidence[0], "", topic,
            caption=f"a contextual image from {topic} that supports one of the premises",
        )
    question = fit_question(question, profile.question_words, rng)
    set_qa(
        qa,
        question=question,
        answer=answer,
        evidence=labeled_evidence,
        family=family_name,
        topic=topic,
    )


def single_hop_fact(
    family: int,
    actor: str,
    topic: str,
    ordinal: int,
    exact: bool,
    rng: random.Random,
) -> tuple[str, str, str, str]:
    color = keyed_choice(COLORS, topic, ordinal, family, "color")
    object_name = keyed_choice(OBJECTS, topic, ordinal, family, "object")
    activity_name = keyed_choice(
        ACTIVITIES, topic, ordinal, family, "activity"
    )
    venue_name = keyed_choice(VENUES, topic, ordinal, family, "venue")
    tag = topic_tag(topic)
    if family == 0:
        answer = with_indefinite_article(f"{color} {object_name}")
        paraphrase = f"the {object_name} with a {color}-colored finish"
        statement = f"The image I shared for {topic} clearly showed {answer if exact else paraphrase}."
        question = rng.choice([
            f"What object appeared in {actor}'s image from {topic}",
            f"Which item was visible in the picture {actor} shared for {topic}",
            f"What did {actor}'s {topic} photograph show",
            f"Name the object pictured in {actor}'s {topic} update",
            f"In {topic}, what appeared in {actor}'s picture",
            f"{actor}'s {topic} photo showed which object",
            f"Identify the item {actor} pictured during {topic}",
            f"Which object can be seen in {actor}'s {topic} image",
        ])
        family_name = "visual-object-recall"
    elif family == 1:
        activity = activity_name
        modifier = ("sunrise", "weekend", "community", "riverside")[ordinal % 4]
        answer = f"{modifier} {activity} at {venue_name}"
        paraphrase = f"practicing {activity} in {venue_name}'s {modifier} sessions"
        statement = f"For {topic}, I began {answer if exact else paraphrase} and kept it in my weekly routine."
        question = rng.choice([
            f"What activity did {actor} begin during {topic}",
            f"Which new practice became part of {actor}'s routine through {topic}",
            f"What did {actor} start doing regularly for {topic}",
            f"Which activity was new for {actor} in the {topic} conversation",
            f"During {topic}, what new activity did {actor} start",
            f"Name the practice {actor} added after {topic}",
            f"{actor} began which routine in the {topic} exchange",
            f"What became a regular {topic} activity for {actor}",
        ])
        family_name = "direct-activity"
    elif family == 2:
        venue = venue_name
        district = keyed_choice(COLORS, topic, ordinal, "district")
        answer = f"{venue}, in the {district} district"
        paraphrase = f"the {venue} located in the district known for {district} signs"
        statement = f"I held the main {topic} meeting at {answer if exact else paraphrase}."
        question = rng.choice([
            f"Where did {actor} hold the main meeting for {topic}",
            f"Which venue hosted {actor}'s {topic} gathering",
            f"What location did {actor} use for {topic}",
            f"At what place did {actor}'s {topic} meeting happen",
            f"During {topic}, where did {actor}'s meeting occur",
            f"Name the venue used by {actor} for {topic}",
            f"{actor}'s {topic} gathering took place where",
            f"Which location appears in {actor}'s {topic} account",
        ])
        family_name = "direct-place"
    elif family == 3:
        reasons = (
            ("a delayed permit", "the permit arrived later than expected"),
            ("storm damage at the venue", "bad weather damaged the meeting space"),
            ("an instructor's illness", "the person leading it became sick"),
            ("a suspended train service", "the trains stopped running that day"),
            ("repairs to the main room", "the room had to be repaired first"),
            ("an understaffed volunteer team", "too few volunteers were available"),
        )
        reason, paraphrase = keyed_choice(reasons, topic, ordinal, "reason")
        answer = f"{reason} during {tag}"
        paraphrase = f"{paraphrase} during the {tag} work"
        statement = f"I postponed {topic} because of {answer if exact else paraphrase}."
        question = rng.choice([
            f"Why did {actor} postpone {topic}",
            f"What caused {actor} to delay {topic}",
            f"Which problem forced the {topic} schedule to change for {actor}",
            f"What reason did {actor} give for moving {topic} to a later date",
            f"During {topic}, what prompted {actor}'s postponement",
            f"Name the obstacle that delayed {actor}'s {topic} plan",
            f"{actor} rescheduled {topic} because of what issue",
            f"Which stated cause moved {actor}'s {topic} date",
        ])
        family_name = "causal-recall"
    elif family == 4:
        answer = with_indefinite_article(f"{color} {object_name}")
        paraphrase = f"the {object_name} finished in {color}"
        statement = f"For {topic}, I bought {answer if exact else paraphrase} after comparing several options."
        question = rng.choice([
            f"What did {actor} buy for {topic}",
            f"Which purchase did {actor} make during {topic}",
            f"What item did {actor} choose for the {topic} work",
            f"What was the object {actor} purchased in connection with {topic}",
            f"During {topic}, what purchase did {actor} make",
            f"Name the item {actor} bought for {topic}",
            f"{actor} chose which object while handling {topic}",
            f"Which bought item appears in {actor}'s {topic} account",
        ])
        family_name = "object-recall"
    elif family == 5:
        feeling_pairs = (
            ("calm and accomplished", "a sense of calm after finishing something worthwhile"),
            ("relieved but optimistic", "relief together with hope about what came next"),
            ("proud of the result", "a strong sense that the finished work was worth celebrating"),
            ("energized and confident", "renewed energy and greater confidence"),
            ("grateful for the support", "thankfulness toward everyone who helped"),
        )
        feeling, paraphrase = keyed_choice(
            feeling_pairs, topic, ordinal, "feeling"
        )
        answer = f"{feeling} about {tag}"
        paraphrase = f"{paraphrase} after the {tag} work"
        statement = f"After {topic}, I felt {answer if exact else paraphrase}."
        question = rng.choice([
            f"How did {actor} feel after {topic}",
            f"What emotion did {actor} associate with completing {topic}",
            f"How did {actor} describe the feeling that followed {topic}",
            f"What was {actor}'s emotional response once {topic} ended",
            f"After {topic}, what feeling did {actor} report",
            f"Name {actor}'s emotional reaction to finishing {topic}",
            f"{actor} felt what way when {topic} ended",
            f"Which emotion followed the {topic} outcome for {actor}",
        ])
        family_name = "emotion-recall"
    elif family == 6:
        number = 3 + ordinal % 12
        activity = activity_name
        answer = f"{number} {tag} {activity} sessions"
        paraphrase = f"finish a set of {activity} meetings totaling {number} sessions"
        statement = f"My main goal for {topic} was to {answer if exact else paraphrase}."
        question = rng.choice([
            f"What goal did {actor} set for {topic}",
            f"What did {actor} aim to complete through {topic}",
            f"Which objective guided {actor}'s work on {topic}",
            f"What outcome was {actor} trying to achieve in {topic}",
            f"During {topic}, which goal was {actor} pursuing",
            f"Name the target {actor} established for {topic}",
            f"{actor}'s {topic} objective was to do what",
            f"Which intended result appears in {actor}'s {topic} plan",
        ])
        family_name = "goal-recall"
    elif family == 7:
        advice_pairs = (
            ("confirm the venue before inviting guests", "check the location first and send invitations afterward"),
            ("divide the work into weekly milestones", "organize the project as smaller goals for each week"),
            ("record each expense in a shared ledger", "keep a common written account of every cost"),
            ("test the route with a smaller group first", "try the route with only a few people before expanding"),
            ("write one observation after every session", "make a short note whenever a session ends"),
        )
        advice, paraphrase = keyed_choice(
            advice_pairs, topic, ordinal, "advice"
        )
        answer = f"{advice} for {tag}"
        paraphrase = f"{paraphrase} for the {tag} work"
        statement = f"The advice I received for {topic} was to {answer if exact else paraphrase}."
        question = rng.choice([
            f"What advice did {actor} receive about {topic}",
            f"Which recommendation was given to {actor} for {topic}",
            f"What practical guidance did {actor} mention in the {topic} update",
            f"What was {actor} advised to do while organizing {topic}",
            f"During {topic}, what guidance reached {actor}",
            f"Name the recommendation {actor} received for {topic}",
            f"{actor} was advised to do what in {topic}",
            f"Which suggested action appears in {actor}'s {topic} account",
        ])
        family_name = "advice-recall"
    elif family == 8:
        answer = f"{color} {object_name} keepsakes"
        paraphrase = f"keepsakes made from {object_name} materials in a {color} shade"
        statement = f"My collection from {topic} consists of {answer if exact else paraphrase}."
        question = rng.choice([
            f"What did {actor} collect during {topic}",
            f"Which keepsakes belong to {actor}'s {topic} collection",
            f"What type of collection did {actor} build through {topic}",
            f"What objects make up the collection {actor} mentioned for {topic}",
            f"During {topic}, which keepsakes did {actor} gather",
            f"Name the objects in {actor}'s {topic} collection",
            f"{actor} collected what kind of items for {topic}",
            f"Which collected objects appear in {actor}'s {topic} account",
        ])
        family_name = "collection-recall"
    elif family == 9:
        trait = keyed_choice(TRAITS, topic, ordinal, "group-trait")
        answer = f"the {color.title()} {trait.title()} Circle"
        paraphrase = f"a circle named for {color} and known for being {trait}"
        statement = f"As part of {topic}, I joined {answer if exact else paraphrase}."
        question = rng.choice([
            f"Which group did {actor} join during {topic}",
            f"What organization did {actor} become a member of through {topic}",
            f"Which circle welcomed {actor} in the {topic} account",
            f"What was the name of the group {actor} joined for {topic}",
            f"During {topic}, what group accepted {actor}",
            f"Name the circle {actor} joined through {topic}",
            f"{actor} became a member of which {topic} organization",
            f"Which membership appears in {actor}'s {topic} account",
        ])
        family_name = "membership-recall"
    elif family == 10:
        local_part = topic_tag(topic).lower()
        answer = f"{local_part}@community.example"
        paraphrase = f"the community mailbox beginning with {local_part}"
        statement = f"The contact email for {topic} was {answer if exact else paraphrase}."
        question = rng.choice([
            f"What email address did {actor} give for {topic}",
            f"Which {topic} mailbox belonged to {actor}'s contact",
            f"During {topic}, what email did {actor} record",
            f"Name the electronic address in {actor}'s {topic} update",
        ])
        family_name = "contact-email-recall"
    elif family == 11:
        digits = 2_000_000 + stable_seed(topic, ordinal, "phone") % 7_999_999
        answer = f"555-{digits // 10_000:03d}-{digits % 10_000:04d}"
        paraphrase = f"the local contact number ending in {digits % 10_000:04d}"
        statement = f"For {topic}, I saved {answer if exact else paraphrase}."
        question = rng.choice([
            f"What phone number did {actor} save for {topic}",
            f"Which contact number appears in {actor}'s {topic} notes",
            f"During {topic}, what number did {actor} record",
            f"Name the telephone contact from {actor}'s {topic} update",
        ])
        family_name = "contact-phone-recall"
    elif family == 12:
        slug = topic_tag(topic).lower()
        answer = f"https://example.org/{slug}"
        paraphrase = f"the example.org page whose path is {slug}"
        statement = f"The reference page for {topic} was {answer if exact else paraphrase}."
        question = rng.choice([
            f"What URL did {actor} share for {topic}",
            f"Which web address appears in {actor}'s {topic} record",
            f"During {topic}, what link did {actor} provide",
            f"Name the website path from {actor}'s {topic} update",
        ])
        family_name = "url-recall"
    elif family == 13:
        weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
        weekday = keyed_choice(weekdays, topic, ordinal, "weekday")
        hour = 8 + stable_seed(topic, ordinal, "schedule-hour") % 9
        answer = f"{weekday} at {hour}:00"
        paraphrase = f"the {weekday} slot beginning when the clock reaches {hour}"
        statement = f"My recurring {topic} session was {answer if exact else paraphrase}."
        question = rng.choice([
            f"When was {actor}'s recurring {topic} session",
            f"Which weekday and time did {actor} reserve for {topic}",
            f"During {topic}, what regular slot did {actor} choose",
            f"Name the scheduled day and hour in {actor}'s {topic} account",
        ])
        family_name = "schedule-slot-recall"
    elif family == 14:
        hour = 1 + stable_seed(topic, ordinal, "clock-hour") % 12
        minute = (0, 15, 30, 45)[stable_seed(topic, ordinal, "minute") % 4]
        answer = f"{hour}:{minute:02d} PM"
        paraphrase = f"quarter-hour slot {minute // 15} after {hour} in the afternoon"
        statement = f"The {topic} check-in began at {answer if exact else paraphrase}."
        question = rng.choice([
            f"What time did {actor}'s {topic} check-in begin",
            f"Which clock time marks {actor}'s {topic} appointment",
            f"During {topic}, when did {actor} start the check-in",
            f"Name the starting time in {actor}'s {topic} update",
        ])
        family_name = "clock-time-recall"
    elif family == 15:
        row = 2 + stable_seed(topic, ordinal, "row") % 18
        status = keyed_choice(
            ("approved", "pending", "reviewed", "scheduled"),
            topic,
            ordinal,
            "row-status",
        )
        answer = f"row {row}: {status}"
        paraphrase = f"the {status} entry positioned on line {row}"
        statement = f"In the {topic} table, {answer if exact else paraphrase}."
        question = rng.choice([
            f"What status was in row {row} of {actor}'s {topic} table",
            f"Which value did {actor} record on {topic} row {row}",
            f"During {topic}, what did table row {row} show",
            f"Name the row-{row} status in {actor}'s {topic} sheet",
        ])
        family_name = "table-row-recall"
    elif family == 16:
        start = 10 + stable_seed(topic, ordinal, "sequence") % 70
        sequence = [start + offset * 3 for offset in range(4)]
        answer = str(sequence[-1])
        paraphrase = f"the fourth value after stepping upward by three from {start}"
        statement = (
            f"The {topic} sequence was {sequence[0]}, {sequence[1]}, "
            f"{sequence[2]}, then {answer if exact else paraphrase}."
        )
        question = rng.choice([
            f"What came after {sequence[2]} in {actor}'s {topic} sequence",
            f"Which final number completed {actor}'s {topic} progression",
            f"During {topic}, what value followed {sequence[2]}",
            f"Name the successor to {sequence[2]} in {actor}'s {topic} list",
        ])
        family_name = "sequence-successor-recall"
    elif family == 17:
        amount = 20 + stable_seed(topic, ordinal, "measurement") % 180
        unit = keyed_choice(
            ("millimeters", "centimeters", "kilograms", "watts"),
            topic,
            ordinal,
            "unit",
        )
        answer = f"{amount} {unit}"
        paraphrase = f"a measured value of {amount} in {unit} units"
        statement = f"The recorded {topic} specification was {answer if exact else paraphrase}."
        question = rng.choice([
            f"What measurement did {actor} record for {topic}",
            f"Which numeric specification appears in {actor}'s {topic} notes",
            f"During {topic}, what measured value did {actor} report",
            f"Name the quantity and unit in {actor}'s {topic} update",
        ])
        family_name = "technical-measurement-recall"
    elif family == 18:
        answer = f"Small steps make {activity_name} sustainable"
        paraphrase = f"steady progress helps the {activity_name} habit last"
        statement = f"My exact {topic} note was, '{answer if exact else paraphrase}.'"
        question = rng.choice([
            f"What exact sentence did {actor} write about {topic}",
            f"Which quotation appears in {actor}'s {topic} note",
            f"During {topic}, what words did {actor} use exactly",
            f"Repeat the short line from {actor}'s {topic} update",
        ])
        family_name = "exact-quotation-recall"
    else:
        person = keyed_choice(PEOPLE, topic, ordinal, "named-person")
        relation = keyed_choice(
            ("mentor", "cousin", "neighbor", "instructor"),
            topic,
            ordinal,
            "named-relation",
        )
        answer = f"{person}, {actor}'s {relation}"
        paraphrase = f"the {relation} named {person}"
        statement = f"For {topic}, I consulted {answer if exact else paraphrase}."
        question = rng.choice([
            f"Who did {actor} consult during {topic}",
            f"Which named {relation} helped {actor} with {topic}",
            f"During {topic}, what person advised {actor}",
            f"Name the {relation} in {actor}'s {topic} update",
        ])
        family_name = "named-person-recall"
    return question, answer, statement, family_name


def generate_single_hop(
    qa: dict,
    profile: QAProfile,
    state: SampleState,
    topic: str,
    ordinal: int,
    rng: random.Random,
) -> None:
    actor = state.speakers[ordinal % 2]
    evidence = state.select_evidence(
        profile.evidence_count,
        profile.evidence_sessions,
        [actor],
        ("single", ordinal),
        caption_required=profile.caption_evidence,
    )
    family = ordinal % 20
    question, answer, statement, family_name = single_hop_fact(
        family,
        actor,
        topic,
        ordinal,
        profile.answer_exact_in_evidence,
        rng,
    )
    for index, dialog_id in enumerate(evidence):
        if profile.caption_only_answer and index == 0:
            state.add_fact(
                dialog_id,
                f"I shared a picture from {topic}, but did not describe its central object in the message.",
                topic,
                caption=f"a clear photograph showing {answer}",
            )
        else:
            support = replace_topic_reference(
                statement, topic, "single", ordinal
            ) if index == 0 else (
                f"A later message about {topic} independently confirmed the same detail."
            )
            state.add_fact(dialog_id, support, topic)
    if profile.caption_evidence and evidence and not profile.caption_only_answer:
        state.add_fact(
            evidence[0], "", topic,
            caption=f"a contextual photograph from {topic} showing the surrounding setting",
        )
    question = fit_question(question, profile.question_words, rng)
    set_qa(
        qa,
        question=question,
        answer=answer,
        evidence=evidence,
        family=family_name,
        topic=topic,
    )


def generate_adversarial(
    qa: dict,
    profile: QAProfile,
    state: SampleState,
    topic: str,
    ordinal: int,
    rng: random.Random,
) -> None:
    actor = state.speakers[ordinal % 2]
    other = state.speakers[1 - ordinal % 2]
    evidence = state.select_evidence(
        profile.evidence_count,
        profile.evidence_sessions,
        [actor],
        ("adversarial", ordinal),
        caption_required=profile.caption_evidence,
    )
    family = ordinal % 20
    known_activity = choose(ACTIVITIES, ordinal)
    if family == 0:
        support = f"For {topic}, I said that I had started {known_activity}."
        question = rng.choice([
            f"What activity did {other} begin during {topic}",
            f"Which {topic} practice was started by {other}",
            f"During {topic}, what new routine did {other} adopt",
            f"Name the activity {other} took up after {topic}",
        ])
        family_name = "adversarial-wrong-speaker"
    elif family == 1:
        support = f"I described enjoying {known_activity} while taking part in {topic}."
        question = rng.choice([
            f"What exact budget did {actor} approve for {topic}",
            f"Which amount was authorized by {actor} during {topic}",
            f"During {topic}, how much money did {actor} approve",
            f"Name the precise {topic} budget assigned by {actor}",
        ])
        family_name = "adversarial-wrong-attribute"
    elif family == 2:
        support = f"I postponed one {topic} meeting and later confirmed that the schedule changed."
        question = rng.choice([
            f"What precise reason caused {actor} to postpone {topic}",
            f"Which stated cause delayed {actor}'s {topic} meeting",
            f"During {topic}, why exactly did {actor} reschedule",
            f"Name the specific obstacle behind {actor}'s {topic} delay",
        ])
        family_name = "adversarial-unsupported-cause"
    elif family == 3:
        support = f"I attended a planning discussion for {topic} but did not record the day."
        question = rng.choice([
            f"On what exact date did {actor} attend the {topic} planning discussion",
            f"Which day records {actor}'s {topic} planning attendance",
            f"During {topic}, when exactly did {actor} join planning",
            f"Name the calendar date of {actor}'s {topic} discussion",
        ])
        family_name = "adversarial-missing-date"
    elif family == 4:
        support = f"For {topic}, I mentioned buying some general supplies without naming a manufacturer."
        question = rng.choice([
            f"Which brand and model did {actor} purchase for {topic}",
            f"What exact product version did {actor} buy during {topic}",
            f"During {topic}, which named model did {actor} select",
            f"Identify the manufacturer and model in {actor}'s {topic} purchase",
        ])
        family_name = "adversarial-missing-specificity"
    elif family == 5:
        support = f"I worked on {topic} independently and did not identify anyone who accompanied me."
        question = rng.choice([
            f"Who accompanied {actor} throughout {topic}",
            f"Which person joined {actor} during {topic}",
            f"During {topic}, who traveled alongside {actor}",
            f"Name {actor}'s companion from the {topic} account",
        ])
        family_name = "adversarial-unsupported-companion"
    elif family == 6:
        support = f"I discussed the goals of {topic}, but never named a street or building number."
        question = rng.choice([
            f"What was the complete street address for {actor}'s {topic} meetings",
            f"Which exact building address hosted {actor}'s {topic} sessions",
            f"During {topic}, where precisely did {actor} meet",
            f"Name the street and number in {actor}'s {topic} record",
        ])
        family_name = "adversarial-missing-location-detail"
    elif family == 7:
        support = f"The {topic} messages described completed work and made no promise about another event."
        question = rng.choice([
            f"What future event did {actor} schedule after {topic}",
            f"Which follow-up gathering was booked by {actor} during {topic}",
            f"After {topic}, what event did {actor} arrange next",
            f"Name the later event promised in {actor}'s {topic} account",
        ])
        family_name = "adversarial-unsupported-future"
    elif family == 8:
        support = f"During {topic}, I referred to {choose(PEOPLE, ordinal)} as a helpful contact without explaining our relationship."
        person = choose(PEOPLE, ordinal)
        question = rng.choice([
            f"How exactly is {person} related to {actor} in the {topic} history",
            f"Which family or work relationship links {person} and {actor} in {topic}",
            f"During {topic}, what role did {person} hold for {actor}",
            f"Name {person}'s precise relationship to {actor} from {topic}",
        ])
        family_name = "adversarial-unsupported-relationship"
    elif family == 9:
        support = f"I said the {topic} outcome was useful, but I did not compare it with a previous year."
        question = rng.choice([
            f"How much more successful was {actor}'s {topic} result than last year's",
            f"What improvement over last year did {actor} achieve in {topic}",
            f"During {topic}, by how much did {actor}'s result increase",
            f"State the year-over-year gain in {actor}'s {topic} outcome",
        ])
        family_name = "adversarial-unsupported-comparison"
    elif family == 10:
        support = f"I shared a photograph from {topic}, although the background was not identified."
        question = rng.choice([
            f"What named landmark appeared behind {actor} in the {topic} photograph",
            f"Which recognizable site was visible in {actor}'s {topic} image",
            f"During {topic}, what landmark stood behind {actor}",
            f"Identify the background monument in {actor}'s {topic} photo",
        ])
        family_name = "adversarial-visual-overreach"
    elif family == 11:
        support = f"I considered joining a course after {topic}, but never said that I enrolled or completed it."
        question = rng.choice([
            f"Which course did {actor} complete after {topic}",
            f"What class did {actor} finish following the {topic} exchange",
            f"After {topic}, which program had {actor} completed",
            f"Name the finished course in {actor}'s {topic} history",
        ])
        family_name = "adversarial-planned-versus-completed"
    elif family == 12:
        support = f"I said the {topic} organizer could be reached online, but gave no address."
        question = rng.choice([
            f"What exact email address did {actor} use for {topic}",
            f"Which mailbox appears in {actor}'s {topic} account",
            f"During {topic}, where could {actor} be emailed",
            f"Name the electronic address recorded by {actor} for {topic}",
        ])
        family_name = "adversarial-unsupported-email"
    elif family == 13:
        support = f"The {topic} notes mention a contact, but include no telephone digits."
        question = rng.choice([
            f"What phone number did {actor} list for {topic}",
            f"Which telephone contact appears in {actor}'s {topic} notes",
            f"During {topic}, what number did {actor} provide",
            f"Name the complete phone contact in {actor}'s {topic} record",
        ])
        family_name = "adversarial-unsupported-phone"
    elif family == 14:
        support = f"I referred to a useful {topic} website without sharing its link."
        question = rng.choice([
            f"What exact URL did {actor} share for {topic}",
            f"Which web address was recorded in {actor}'s {topic} history",
            f"During {topic}, what link did {actor} provide",
            f"Name the website path from {actor}'s {topic} account",
        ])
        family_name = "adversarial-unsupported-url"
    elif family == 15:
        support = f"The {topic} exchange summarized {actor}'s view without preserving exact wording."
        question = rng.choice([
            f"What exact sentence did {actor} say during {topic}",
            f"Which verbatim quotation appears in {actor}'s {topic} record",
            f"During {topic}, what were {actor}'s precise words",
            f"Repeat the exact line attributed to {actor} in {topic}",
        ])
        family_name = "adversarial-unsupported-quotation"
    elif family == 16:
        support = f"I said that several people joined {topic}, but no attendance total was recorded."
        question = rng.choice([
            f"How many people exactly attended {actor}'s {topic} event",
            f"Which final attendance count appears in {actor}'s {topic} record",
            f"During {topic}, what was {actor}'s precise participant total",
            f"State the exact number of attendees reported by {actor} for {topic}",
        ])
        family_name = "adversarial-unsupported-count"
    elif family == 17:
        support = f"Early in {topic}, I considered {known_activity}, but never confirmed it as my current preference."
        question = rng.choice([
            f"What is {actor}'s current favorite activity after {topic}",
            f"Which present-day preference did {actor} confirm in {topic}",
            f"After {topic}, what pastime does {actor} now prefer",
            f"Name the active preference in {actor}'s latest {topic} update",
        ])
        family_name = "adversarial-stale-versus-current"
    elif family == 18:
        support = f"The {topic} discussion mentioned meeting again, without setting a day or time."
        question = rng.choice([
            f"When exactly was {actor}'s next {topic} meeting scheduled",
            f"Which weekday and time did {actor} reserve after {topic}",
            f"During {topic}, what precise future slot did {actor} choose",
            f"Name the confirmed date and hour in {actor}'s {topic} plan",
        ])
        family_name = "adversarial-unsupported-schedule"
    else:
        support = (
            f"I referred to {with_indefinite_article(f'{topic} table')}, "
            "but never described its columns or cell values."
        )
        question = rng.choice([
            f"What value was in row five of {actor}'s {topic} table",
            f"Which status appears in {actor}'s fifth {topic} row",
            f"During {topic}, what did table cell B5 contain",
            f"Name the row-five entry from {actor}'s {topic} sheet",
        ])
        family_name = "adversarial-unsupported-table-field"

    for index, dialog_id in enumerate(evidence):
        state.add_fact(
            dialog_id,
            support if index == 0 else f"Another {topic} message stayed silent on the requested detail.",
            topic,
        )
    if profile.caption_evidence and evidence:
        state.add_fact(
            evidence[0], "", topic,
            caption=f"a contextual photograph from {topic} without the requested identifying detail",
        )
    question = fit_question(question, profile.question_words, rng)
    set_qa(
        qa,
        question=question,
        answer=None,
        evidence=evidence,
        family=family_name,
        topic=topic,
    )


def clip_words(text: str, count: int) -> str:
    if count <= 0:
        return ""
    matches = list(WORD_RE.finditer(text))
    if len(matches) <= count:
        return text.strip()
    return text[: matches[count - 1].end()].rstrip(" ,;:") + "."


def compose_turn(required: str, background: str, target_words: int) -> str:
    required = required.strip()
    target_words = max(1, target_words)
    if not required:
        result = clip_words(background, target_words)
        return result
    remaining = max(0, target_words - word_count(required))
    if remaining == 0:
        return required
    filler = clip_words(background, remaining)
    result = f"{required} {filler}".strip()
    return clip_words(result, target_words)


def render_conversations(
    dataset: list[dict],
    official_targets: list[list[int]],
    states: list[SampleState],
    background: BackgroundProvider,
    topics: list[str],
) -> None:
    for sample_index, (sample, state) in enumerate(zip(dataset, states)):
        targets = iter(official_targets[sample_index])
        for session_number in sorted(state.sessions):
            session_topic = topics[
                (sample_index * 17 + session_number * 7) % len(topics)
            ]
            for turn in state.sessions[session_number]:
                target = next(targets)
                stored_target = int(turn.pop("_target_words"))
                if stored_target != target:
                    raise ValueError("Turn word-target alignment changed during generation")
                required = " ".join(turn.pop("_facts"))
                turn_topics = turn.pop("_topics")
                topic = turn_topics[0] if turn_topics else session_topic
                role = (
                    "user"
                    if turn.get("speaker") == sample["conversation"]["speaker_a"]
                    else "assistant"
                )
                requested = max(8, target - word_count(required))
                fragment = background.take_topic_sentence_fragments(
                    topic,
                    role,
                    [requested],
                )[0]
                supplemental = background.take_sentence_fragments(
                    role,
                    [max(12, target)],
                )[0]
                turn["text"] = compose_turn(
                    required,
                    f"{fragment} {supplemental}",
                    target,
                )


def leakage_report(official: list[dict], synthetic: list[dict]) -> dict:
    official_questions = {
        normalized(qa.get("question"))
        for sample in official
        for qa in sample.get("qa") or []
    }
    official_long_answers = {
        normalized(qa.get("answer"))
        for sample in official
        for qa in sample.get("qa") or []
        if word_count(qa.get("answer")) >= 5
    }
    official_turns = {
        normalized(turn.get("text"))
        for sample in official
        for key, session in sample.get("conversation", {}).items()
        if SESSION_RE.fullmatch(key) and isinstance(session, list)
        for turn in session
        if word_count(turn.get("text")) >= 5
    }
    synthetic_questions = [
        normalized(qa.get("question"))
        for sample in synthetic
        for qa in sample.get("qa") or []
    ]
    synthetic_long_answers = [
        normalized(qa.get("answer"))
        for sample in synthetic
        for qa in sample.get("qa") or []
        if word_count(qa.get("answer")) >= 5
    ]
    synthetic_turns = [
        normalized(turn.get("text"))
        for sample in synthetic
        for key, session in sample.get("conversation", {}).items()
        if SESSION_RE.fullmatch(key) and isinstance(session, list)
        for turn in session
        if word_count(turn.get("text")) >= 5
    ]
    return {
        "exact_official_question_overlap_count": sum(
            value in official_questions for value in synthetic_questions
        ),
        "exact_official_long_answer_overlap_count": sum(
            value in official_long_answers for value in synthetic_long_answers
        ),
        "exact_official_turn_overlap_count_min_5_words": sum(
            value in official_turns for value in synthetic_turns
        ),
        "unique_generated_question_count": len(set(synthetic_questions)),
    }


def validate_dataset(dataset: list[dict]) -> dict:
    category_counts: Counter[int] = Counter()
    question_ids = []
    questions = []
    errors = []
    for sample in dataset:
        dialog_ids = {
            str(turn.get("dia_id"))
            for key, session in sample["conversation"].items()
            if SESSION_RE.fullmatch(key) and isinstance(session, list)
            for turn in session
        }
        for qa in sample.get("qa") or []:
            category = int(qa["category"])
            category_counts[category] += 1
            question_ids.append(str(qa.get("question_id")))
            questions.append(normalized(qa.get("question")))
            evidence = {str(item) for item in qa.get("evidence") or []}
            if not evidence.issubset(dialog_ids):
                errors.append(f"invalid evidence for {qa.get('question_id')}")
            if category == 5 and qa.get("answer") is not None:
                errors.append(f"category 5 answer is not null for {qa.get('question_id')}")
            if category != 5 and qa.get("answer") in {None, ""}:
                errors.append(f"missing answer for {qa.get('question_id')}")
    if len(set(question_ids)) != len(question_ids):
        errors.append("question IDs are not unique")
    if len(set(questions)) != len(questions):
        errors.append("question strings are not unique")
    if dict(sorted(category_counts.items())) != v1.EXPECTED_CATEGORY_COUNTS:
        errors.append("category counts do not match official LoCoMo")
    if errors:
        raise ValueError("Synthetic LoCoMo V2 validation failed:\n- " + "\n- ".join(errors[:20]))
    return {
        "conversation_count": len(dataset),
        "question_count": sum(category_counts.values()),
        "category_counts": dict(sorted(category_counts.items())),
        "unique_question_count": len(set(questions)),
    }


def generate_dataset(
    official: list[dict],
    topics: list[str],
    background: BackgroundProvider,
    seed: int,
) -> tuple[list[dict], dict]:
    previous_label_for = v1.label_for
    v1.label_for = lambda ordinal: topics[ordinal % len(topics)]
    try:
        dataset = v1.generate_dataset(seed=seed)
    finally:
        v1.label_for = previous_label_for

    profiles = official_profiles(official)
    turn_targets = official_turn_word_targets(official)
    states = [
        SampleState(
            sample,
            turn_targets[sample_index],
            stable_seed(seed, sample_index),
        )
        for sample_index, sample in enumerate(dataset)
    ]
    for sample_index, (sample, state) in enumerate(zip(dataset, states)):
        category_offsets: Counter[int] = Counter()
        for qa_index, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            ordinal = category_offsets[category]
            category_offsets[category] += 1
            profile = profiles[sample_index][category][ordinal]
            topic = topics[
                (sample_index * 31 + category * 17 + ordinal * 37) % len(topics)
            ]
            rng = random.Random(stable_seed(seed, sample_index, category, ordinal))
            qa["question_id"] = str(qa["question_id"]).replace(
                "synthetic_locomo_", "synthetic_locomo_v2_", 1
            )
            if category == 1:
                generate_multi_hop(qa, profile, state, topic, ordinal, rng)
            elif category == 2:
                generate_temporal(qa, profile, state, topic, ordinal, rng)
            elif category == 3:
                generate_open_domain(qa, profile, state, topic, ordinal, rng)
            elif category == 4:
                generate_single_hop(qa, profile, state, topic, ordinal, rng)
            else:
                generate_adversarial(qa, profile, state, topic, ordinal, rng)

        for qa in sample["qa"]:
            if int(qa["category"]) != 5:
                state.add_decoys(
                    qa["question"],
                    qa.get("answer"),
                    qa["_topic"],
                    [str(item) for item in qa.get("evidence") or []],
                    count=2,
                )
            qa.pop("_topic", None)
        sample["synthetic_metadata"] = {
            "version": 2,
            "purpose": "high_fidelity_development_and_regression_testing_only",
            "official_score": False,
            "official_content_emitted": False,
        }

    render_conversations(
        dataset,
        turn_targets,
        states,
        background,
        topics,
    )
    return dataset, validate_dataset(dataset)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-data",
        type=Path,
        default=Path("benchmark/locomo/official/locomo10.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark/locomo/synthetic_v2/"
            "pragmos_synthetic_locomo_1986_v2.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "benchmark/locomo/synthetic_v2/"
            "pragmos_synthetic_locomo_1986_v2_manifest.json"
        ),
    )
    parser.add_argument(
        "--semantic-report",
        type=Path,
        default=Path(
            "benchmark/locomo/synthetic_v2/"
            "pragmos_synthetic_locomo_1986_v2_semantic_report.json"
        ),
    )
    parser.add_argument(
        "--background-user",
        type=Path,
        default=Path(
            "benchmark/longMemEval/synthetic_v2/background/"
            "ultrachat_user_background.jsonl"
        ),
    )
    parser.add_argument(
        "--background-assistant",
        type=Path,
        default=Path(
            "benchmark/longMemEval/synthetic_v2/background/"
            "ultrachat_assistant_background.jsonl"
        ),
    )
    parser.add_argument(
        "--background-topics",
        type=Path,
        default=Path(
            "benchmark/longMemEval/synthetic_v2/background/"
            "ultrachat_topic_phrases.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    for path in (
        args.official_data,
        args.background_user,
        args.background_assistant,
        args.background_topics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    official = json.loads(args.official_data.read_text(encoding="utf-8"))
    raw_topics = json.loads(args.background_topics.read_text(encoding="utf-8"))
    topics = [
        topic
        for topic in raw_topics
        if 2 <= word_count(topic) <= 4
        and not any(character.isdigit() for character in topic)
    ][:113]
    if len(topics) < 97:
        raise ValueError("Insufficient independent topic phrases")
    random.Random(args.seed).shuffle(topics)
    topics = topics[:97]
    background = BackgroundProvider(
        args.background_user,
        args.background_assistant,
    )
    background.index_topic_contexts(topics)
    topic_context_summary = background.topic_context_summary(topics)
    if topic_context_summary["topics_with_both_roles"] != len(topics):
        raise ValueError(
            "Incomplete independent topic context: "
            + json.dumps(topic_context_summary, sort_keys=True)
        )
    try:
        dataset, validation = generate_dataset(
            official,
            topics,
            background,
            args.seed,
        )
    finally:
        background.close()

    leakage = leakage_report(official, dataset)
    if any(
        leakage[key]
        for key in (
            "exact_official_question_overlap_count",
            "exact_official_long_answer_overlap_count",
            "exact_official_turn_overlap_count_min_5_words",
        )
    ):
        raise ValueError("Official-content leakage detected: " + json.dumps(leakage))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(dataset))

    official_metrics = analyze_dataset(official)
    synthetic_metrics = analyze_dataset(dataset)
    comparison = compare_reports(official_metrics, synthetic_metrics)
    semantic_report = {
        "schema_version": 1,
        "official_file": str(args.official_data),
        "official_sha256": sha256_path(args.official_data),
        "synthetic_file": str(args.output),
        "synthetic_sha256": sha256_path(args.output),
        "official": official_metrics,
        "synthetic": synthetic_metrics,
        "comparison": comparison,
    }
    args.semantic_report.write_text(
        json.dumps(semantic_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "dataset_name": "PRAGMOS Synthetic LoCoMo Development Set V2",
        "purpose": "high_fidelity_development_and_regression_testing_only",
        "official_locomo_score": False,
        "independent_evaluation_set": False,
        "seed": args.seed,
        "generator": Path(__file__).name,
        "generator_sha256": sha256_path(Path(__file__)),
        "dataset_file": args.output.name,
        "dataset_sha256": sha256_path(args.output),
        "semantic_report_file": args.semantic_report.name,
        "semantic_report_sha256": sha256_path(args.semantic_report),
        "official_profile_source": str(args.official_data),
        "official_profile_sha256": sha256_path(args.official_data),
        "official_profile_commit": OFFICIAL_COMMIT,
        "official_profile_fields_used": [
            "category counts", "question and answer lengths",
            "evidence-turn and evidence-session counts", "caption prevalence",
            "answer extractiveness", "turn-length profile",
            "aggregate retrieval and lexical-difficulty diagnostics",
        ],
        "official_content_never_emitted": [
            "questions", "answers", "conversation turns", "speaker names",
            "dates", "session summaries", "event summaries", "observations",
        ],
        "independent_background": {
            "user_path": str(args.background_user),
            "assistant_path": str(args.background_assistant),
            "topic_path": str(args.background_topics),
            "topic_context_summary": topic_context_summary,
        },
        "validation": validation,
        "leakage_checks": leakage,
        "difficulty_comparison_passed": comparison["passed"],
        "publication_note": (
            "V2 is benchmark-shaped development data, not an independent or "
            "official LoCoMo result. Only untouched official inference runs may "
            "be reported as official benchmark performance."
        ),
    }
    args.manifest.write_bytes(json_bytes(manifest))
    print(f"Wrote dataset: {args.output}")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote semantic report: {args.semantic_report}")
    print(json.dumps({
        "validation": validation,
        "leakage": leakage,
        "difficulty_comparison": comparison,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
