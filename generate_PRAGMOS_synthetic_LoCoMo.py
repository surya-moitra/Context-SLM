#!/usr/bin/env python3
"""Generate an independent LoCoMo-shaped development and regression dataset.

Only the public LoCoMo schema and aggregate category/session counts are mirrored.
Official conversations, questions, answers, evidence text, entities, and dates are
not inputs to generation.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_SEED = 20261003
OFFICIAL_REPOSITORY = "https://github.com/snap-research/LoCoMo"
OFFICIAL_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
EXPECTED_CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}


@dataclass(frozen=True)
class ConversationProfile:
    sample_id: str
    speaker_a: str
    speaker_b: str
    session_count: int
    target_turn_count: int
    category_counts: dict[int, int]


CONVERSATION_PROFILES = (
    ConversationProfile("synthetic-locomo-01", "Asha", "Rohan", 19, 419, {1: 32, 2: 37, 3: 13, 4: 70, 5: 47}),
    ConversationProfile("synthetic-locomo-02", "Elena", "Marcus", 19, 369, {1: 11, 2: 26, 3: 0, 4: 44, 5: 24}),
    ConversationProfile("synthetic-locomo-03", "Noor", "Diego", 32, 663, {1: 31, 2: 27, 3: 8, 4: 86, 5: 41}),
    ConversationProfile("synthetic-locomo-04", "Priyanka", "Felix", 29, 629, {1: 37, 2: 40, 3: 11, 4: 111, 5: 61}),
    ConversationProfile("synthetic-locomo-05", "Lila", "Omar", 29, 680, {1: 31, 2: 26, 3: 14, 4: 107, 5: 64}),
    ConversationProfile("synthetic-locomo-06", "Hana", "Victor", 28, 675, {1: 30, 2: 24, 3: 7, 4: 62, 5: 35}),
    ConversationProfile("synthetic-locomo-07", "Mira", "Leon", 31, 689, {1: 20, 2: 34, 3: 13, 4: 83, 5: 40}),
    ConversationProfile("synthetic-locomo-08", "Celeste", "Arjun", 30, 681, {1: 21, 2: 42, 3: 10, 4: 118, 5: 48}),
    ConversationProfile("synthetic-locomo-09", "Sora", "Matteo", 25, 509, {1: 37, 2: 33, 3: 13, 4: 73, 5: 40}),
    ConversationProfile("synthetic-locomo-10", "Inez", "Kiran", 30, 568, {1: 32, 2: 32, 3: 7, 4: 87, 5: 46}),
)

ADJECTIVES = (
    "Amber",
    "Cobalt",
    "Copper",
    "Emerald",
    "Indigo",
    "Juniper",
    "Maple",
    "Silver",
    "Willow",
    "Violet",
)
NOUNS = (
    "Atlas",
    "Bridge",
    "Compass",
    "Garden",
    "Harbor",
    "Lantern",
    "Meadow",
    "Orchard",
    "Studio",
    "Workshop",
)
PROJECT_PHASES = (
    "autumn program",
    "community initiative",
    "evening series",
    "field study",
    "neighborhood plan",
    "open-house effort",
    "pilot program",
    "public showcase",
    "seasonal project",
    "skills exchange",
    "spring program",
    "summer series",
    "winter project",
)
ACTIVITIES = (
    "ceramics",
    "birdwatching",
    "watercolor painting",
    "bread baking",
    "trail running",
    "calligraphy",
    "salsa dancing",
    "wood carving",
    "urban gardening",
    "kayaking",
)
PLACES = (
    "Cedar Point Library",
    "Harbor Arts Center",
    "Juniper Community Hall",
    "Maple Street Studio",
    "Northwind Museum",
    "Orchid Learning Center",
    "Pioneer Theatre",
    "Riverside Workshop",
    "Silver Lake Park",
    "Willow Market",
)
ITEMS = (
    "a cobalt travel journal",
    "a compact brass telescope",
    "a forest-green rain jacket",
    "a hand-thrown ceramic bowl",
    "a maple-wood chess set",
    "a portable watercolor kit",
    "a silver fountain pen",
    "a solar camping lantern",
    "an ivory recipe binder",
    "an orange field recorder",
)
REASONS = (
    "the venue lost power",
    "heavy rain closed the trail",
    "the instructor became ill",
    "the delivery arrived late",
    "the permit was delayed",
    "the room needed repairs",
    "the train service was suspended",
    "the volunteer team was understaffed",
)
FEELINGS = (
    "calm and accomplished",
    "confident",
    "energized",
    "grateful",
    "hopeful",
    "proud",
    "relieved",
)
GOALS = (
    "finish twelve practice sessions",
    "raise funds for new art supplies",
    "teach three beginner workshops",
    "complete the route before sunset",
    "publish a neighborhood guide",
    "organize a community exhibition",
)
ADVICE = (
    "break the work into weekly milestones",
    "confirm the venue before inviting guests",
    "practice slowly before increasing speed",
    "record each expense in a shared ledger",
    "test the route with a smaller group first",
    "write down one observation after every session",
)
COLLECTIONS = (
    "botanical postcards",
    "concert tickets",
    "handmade bookmarks",
    "local history pamphlets",
    "painted pebbles",
    "railway maps",
    "recipe cards",
    "vintage buttons",
)
COUNTRY_BY_CITY = {
    "Kyoto": "Japan",
    "Lisbon": "Portugal",
    "Nairobi": "Kenya",
    "Oslo": "Norway",
    "Quito": "Ecuador",
    "Seoul": "South Korea",
}
STATE_BY_CITY = {
    "Austin": "Texas",
    "Denver": "Colorado",
    "Portland": "Oregon",
    "Seattle": "Washington",
    "Tucson": "Arizona",
}
COMMUNITY_ACTIONS = (
    "served meals at the neighborhood pantry",
    "sorted winter coats for the shelter",
    "delivered library books to homebound residents",
    "planted trees beside the community center",
    "tutored students after school",
    "organized supplies for the food bank",
    "cleaned litter from the riverside path",
    "prepared care packages for hospital patients",
    "collected school materials for new families",
    "helped repair bicycles for local commuters",
    "translated forms at the community clinic",
    "coordinated rides for older residents",
    "staffed the neighborhood recycling drive",
    "read stories at the children's library",
    "assembled hygiene kits for the shelter",
    "maintained the shared vegetable garden",
    "mentored volunteers at the youth center",
)


def label_for(ordinal: int) -> str:
    adjective = ADJECTIVES[ordinal % len(ADJECTIVES)]
    noun = NOUNS[(ordinal // len(ADJECTIVES)) % len(NOUNS)]
    phase = PROJECT_PHASES[(ordinal // 100) % len(PROJECT_PHASES)]
    return f"{adjective} {noun} {phase}"


def format_session_datetime(value: datetime) -> str:
    hour = value.strftime("%I").lstrip("0") or "0"
    return f"{hour}:{value:%M %p}".lower() + f" on {value.day} {value:%B, %Y}"


def format_date(value: datetime) -> str:
    return f"{value.day} {value:%B %Y}"


def previous_weekday(value: datetime, weekday: int) -> datetime:
    days_back = (value.weekday() - weekday) % 7
    if days_back == 0:
        days_back = 7
    return value - timedelta(days=days_back)


def add_month(value: datetime) -> datetime:
    year = value.year + (1 if value.month == 12 else 0)
    month = 1 if value.month == 12 else value.month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


class ConversationBuilder:
    def __init__(self, profile: ConversationProfile, rng: random.Random, index: int):
        self.profile = profile
        self.rng = rng
        self.index = index
        self.start = datetime(2027 + (index % 2), 1 + (index % 5), 4, 10, 15)
        self.session_dates = {
            session: self.start
            + timedelta(days=(session - 1) * 8 + (session % 3), hours=session % 7)
            for session in range(1, profile.session_count + 1)
        }
        self.sessions = {
            session: [] for session in range(1, profile.session_count + 1)
        }

    @property
    def speakers(self) -> tuple[str, str]:
        return self.profile.speaker_a, self.profile.speaker_b

    def other(self, speaker: str) -> str:
        return self.profile.speaker_b if speaker == self.profile.speaker_a else self.profile.speaker_a

    def session_for(self, ordinal: int, offset: int = 0) -> int:
        return 1 + ((ordinal * 7 + offset * 5 + self.index * 3) % self.profile.session_count)

    def add_exchange(
        self,
        session: int,
        speaker: str,
        text: str,
        *,
        caption: str | None = None,
        reply: bool = True,
    ) -> str:
        turns = self.sessions[session]
        dia_id = f"D{session}:{len(turns) + 1}"
        turn = {"speaker": speaker, "dia_id": dia_id, "text": text}
        if caption:
            turn["blip_caption"] = caption
            turn["synthetic_media"] = True
        turns.append(turn)
        if reply:
            turns.append(
                {
                    "speaker": self.other(speaker),
                    "dia_id": f"D{session}:{len(turns) + 1}",
                    "text": "Thanks for the update. I will keep that detail in mind.",
                }
            )
        return dia_id

    def add_noise(self) -> None:
        noise_topics = (
            "sorted old receipts by month",
            "watered the balcony herbs",
            "charged the bicycle lights",
            "returned two library books",
            "cleaned the kitchen shelves",
            "checked tomorrow's weather forecast",
        )
        for session in range(1, self.profile.session_count + 1):
            speaker = self.speakers[session % 2]
            topic = noise_topics[(session + self.index) % len(noise_topics)]
            self.add_exchange(
                session,
                speaker,
                f"A small unrelated update: I {topic}. Nothing else changed.",
            )
        filler_ordinal = 0
        while self.turn_count() + 2 <= self.profile.target_turn_count:
            session = 1 + (filler_ordinal % self.profile.session_count)
            speaker = self.speakers[(filler_ordinal + session) % 2]
            topic = noise_topics[(filler_ordinal + self.index) % len(noise_topics)]
            self.add_exchange(
                session,
                speaker,
                f"Another unrelated note: I {topic}. This did not alter any plans.",
            )
            filler_ordinal += 1
        if self.turn_count() < self.profile.target_turn_count:
            session = 1 + (filler_ordinal % self.profile.session_count)
            self.sessions[session].append(
                {
                    "speaker": self.speakers[filler_ordinal % 2],
                    "dia_id": f"D{session}:{len(self.sessions[session]) + 1}",
                    "text": "That concludes the unrelated housekeeping notes for today.",
                }
            )

    def turn_count(self) -> int:
        return sum(len(turns) for turns in self.sessions.values())

    def as_conversation(self) -> dict:
        conversation = {
            "speaker_a": self.profile.speaker_a,
            "speaker_b": self.profile.speaker_b,
        }
        for session in range(1, self.profile.session_count + 1):
            conversation[f"session_{session}"] = self.sessions[session]
            conversation[f"session_{session}_date_time"] = format_session_datetime(
                self.session_dates[session]
            )
        return conversation


def qa_record(
    sample_index: int,
    category: int,
    ordinal: int,
    question: str,
    answer,
    evidence: list[str],
    family: str,
) -> dict:
    return {
        "question_id": (
            f"synthetic_locomo_{sample_index + 1:02d}_c{category}_{ordinal + 1:04d}"
        ),
        "question": question,
        "answer": answer,
        "category": category,
        "evidence": evidence,
        "synthetic_family": family,
    }


def generate_single_hop(
    builder: ConversationBuilder,
    count: int,
    sample_index: int,
) -> tuple[list[dict], list[dict]]:
    rows = []
    adversarial_specs = []
    for ordinal in range(count):
        speaker = builder.speakers[(ordinal + sample_index) % 2]
        other = builder.other(speaker)
        label = label_for(sample_index * 200 + ordinal)
        family = ordinal % 10
        answer = None
        caption = None
        if ordinal % 8 == 0:
            answer = ITEMS[(ordinal + sample_index) % len(ITEMS)]
            question = f"What object appeared in {speaker}'s photo for the {label}?"
            text = f"I finally documented the {label}. Here is the photo I mentioned."
            caption = f"a clear photo showing {answer} prepared for the {label}"
            family_name = "visual-caption-single-hop"
        elif family == 1:
            answer = ACTIVITIES[(ordinal + sample_index) % len(ACTIVITIES)]
            question = f"What activity did {speaker} begin for the {label}?"
            text = f"For the {label}, I began {answer} and now practice every weekend."
            family_name = "direct-activity"
        elif family == 2:
            answer = PLACES[(ordinal + sample_index) % len(PLACES)]
            question = f"Where did {speaker} hold the {label}?"
            text = f"I held the {label} at {answer}; the main room had enough space."
            family_name = "direct-place"
        elif family == 3:
            answer = REASONS[(ordinal + sample_index) % len(REASONS)]
            question = f"Why did {speaker} postpone the {label}?"
            text = f"I postponed the {label} because {answer}."
            family_name = "causal-recall"
        elif family == 4:
            answer = ITEMS[(ordinal + sample_index) % len(ITEMS)]
            question = f"What did {speaker} buy for the {label}?"
            text = f"For the {label}, I bought {answer} after comparing several options."
            family_name = "object-recall"
        elif family == 5:
            answer = FEELINGS[(ordinal + sample_index) % len(FEELINGS)]
            question = f"How did {speaker} feel after the {label}?"
            text = f"After the {label}, I felt {answer}."
            family_name = "emotion-recall"
        elif family == 6:
            answer = GOALS[(ordinal + sample_index) % len(GOALS)]
            question = f"What goal did {speaker} set for the {label}?"
            text = f"My goal for the {label} is to {answer}."
            family_name = "goal-recall"
        elif family == 7:
            answer = ADVICE[(ordinal + sample_index) % len(ADVICE)]
            question = f"What advice did {speaker} receive about the {label}?"
            text = f"My neighbor advised me to {answer} for the {label}."
            family_name = "advice-recall"
        elif family == 8:
            answer = COLLECTIONS[(ordinal + sample_index) % len(COLLECTIONS)]
            question = f"What did {speaker} collect during the {label}?"
            text = f"During the {label}, I collected {answer}."
            family_name = "collection-recall"
        else:
            answer = f"team {ADJECTIVES[(ordinal + sample_index) % len(ADJECTIVES)]}"
            question = f"Which team did {speaker} join for the {label}?"
            text = f"I joined {answer} for the {label}."
            family_name = "membership-recall"

        if caption is None and ordinal % 3 == 0:
            caption = f"a contextual photo taken during the {label}"

        session = builder.session_for(ordinal, family)
        evidence_id = builder.add_exchange(
            session,
            speaker,
            text,
            caption=caption,
        )
        rows.append(
            qa_record(
                sample_index,
                4,
                ordinal,
                question,
                answer,
                [evidence_id],
                family_name,
            )
        )
        if ordinal % 3 == 2:
            adversarial_question = (
                f"What budget did {speaker} approve for the {label}?"
            )
            adversarial_family = "adversarial-wrong-attribute"
        else:
            adversarial_question = question.replace(speaker, other)
            adversarial_family = "adversarial-wrong-speaker"
        adversarial_specs.append(
            {
                "question": adversarial_question,
                "evidence": [evidence_id],
                "family": adversarial_family,
            }
        )
    return rows, adversarial_specs


def generate_multi_hop(
    builder: ConversationBuilder,
    count: int,
    sample_index: int,
) -> list[dict]:
    rows = []
    for ordinal in range(count):
        speaker = builder.speakers[(ordinal + sample_index) % 2]
        other = builder.other(speaker)
        label = label_for(3000 + sample_index * 100 + ordinal)
        family = ordinal % 8
        evidence = []
        if family == 0:
            values = [ACTIVITIES[(ordinal + step + sample_index) % len(ACTIVITIES)] for step in range(3)]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"For the {label}, I added {value} to my activities.",
                        caption=(
                            f"a photo from the {label} activity"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"What activities did {speaker} add during the {label}?"
            answer = ", ".join(values)
            family_name = "cross-session-list-union"
        elif family == 1:
            values = [PLACES[(ordinal + step + sample_index) % len(PLACES)] for step in range(4)]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"Another stop in my {label} visits was {value}.",
                        caption=(
                            f"a travel photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"Which places did {speaker} visit for the {label}?"
            answer = ", ".join(values)
            family_name = "cross-session-place-union"
        elif family == 2:
            values = [f"workshop {ADJECTIVES[(ordinal + step) % len(ADJECTIVES)]}" for step in range(3)]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"I attended {value} as part of the {label} series.",
                        caption=(
                            f"a workshop photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"How many distinct workshops did {speaker} attend in the {label} series?"
            answer = "3"
            family_name = "cross-session-count-distinct"
        elif family == 3:
            common = ACTIVITIES[(ordinal + sample_index) % len(ACTIVITIES)]
            for step, participant in enumerate((speaker, other)):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        participant,
                        f"For the {label}, one activity I especially enjoy is {common}.",
                        caption=(
                            f"a shared-activity photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"What activity do {speaker} and {other} both enjoy in the {label}?"
            answer = common
            family_name = "speaker-set-intersection"
        elif family == 4:
            values = [COLLECTIONS[(ordinal + step + sample_index) % len(COLLECTIONS)] for step in range(3)]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"My {label} archive now includes {value}.",
                        caption=(
                            f"an archive photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"What collections are included in {speaker}'s {label} archive?"
            answer = ", ".join(values)
            family_name = "cross-session-collection-union"
        elif family == 5:
            values = [GOALS[(ordinal + step + sample_index) % len(GOALS)] for step in range(2)]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"One goal in my {label} plan is to {value}.",
                        caption=(
                            f"a planning photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"What goals did {speaker} set in the {label} plan?"
            answer = ", ".join(values)
            family_name = "cross-session-goal-union"
        elif family == 6:
            values = [ITEMS[(ordinal + step + sample_index) % len(ITEMS)] for step in range(3)]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"I received {value} as a recommendation for the {label}.",
                        caption=(
                            f"a recommendation photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"What items were recommended to {speaker} for the {label}?"
            answer = ", ".join(values)
            family_name = "cross-session-recommendation-union"
        else:
            item_count = 8 + ((ordinal + sample_index) % 12)
            values = [
                (
                    f"{ACTIVITIES[(ordinal + step) % len(ACTIVITIES)]} at "
                    f"{PLACES[(ordinal + (step // len(ACTIVITIES)) * 3) % len(PLACES)]}"
                )
                for step in range(item_count)
            ]
            for step, value in enumerate(values):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        f"My {label} log includes {value}.",
                        caption=(
                            f"a log photo from the {label}"
                            if step == 0 and ordinal % 10 < 7
                            else None
                        ),
                        reply=False,
                    )
                )
            question = f"What entries did {speaker} record throughout the {label}?"
            answer = ", ".join(values)
            family_name = "long-history-list-union"
        rows.append(
            qa_record(
                sample_index,
                1,
                ordinal,
                question,
                answer,
                evidence,
                family_name,
            )
        )
    return rows


def generate_temporal(
    builder: ConversationBuilder,
    count: int,
    sample_index: int,
) -> list[dict]:
    rows = []
    for ordinal in range(count):
        speaker = builder.speakers[(ordinal + sample_index) % 2]
        label = label_for(6000 + sample_index * 100 + ordinal)
        session = builder.session_for(ordinal, 2)
        when = builder.session_dates[session]
        family = ordinal % 9
        evidence_ids = []
        if family == 0:
            event_date = when - timedelta(days=1)
            text = f"Yesterday I completed the final review for the {label}."
            question = f"When did {speaker} complete the final review for the {label}?"
            answer = format_date(event_date)
            family_name = "relative-yesterday"
        elif family == 1:
            text = f"Last week I attended the planning meeting for the {label}."
            question = f"When did {speaker} attend the planning meeting for the {label}?"
            answer = f"The week before {format_date(when)}"
            family_name = "relative-last-week"
        elif family == 2:
            event_date = add_month(when)
            text = f"Next month I plan to launch the {label}."
            question = f"When is {speaker} planning to launch the {label}?"
            answer = event_date.strftime("%B %Y")
            family_name = "relative-next-month"
        elif family == 3:
            event_date = when - timedelta(days=3)
            text = f"Three days ago I signed the venue form for the {label}."
            question = f"When did {speaker} sign the venue form for the {label}?"
            answer = format_date(event_date)
            family_name = "relative-days-ago"
        elif family == 4:
            years = 2 + ((ordinal + sample_index) % 6)
            text = f"I have maintained the {label} archive for {years} years."
            question = f"How long has {speaker} maintained the {label} archive?"
            answer = f"{years} years"
            family_name = "explicit-duration"
        elif family == 5:
            event_date = previous_weekday(when, 4)
            text = f"Last Friday I presented the {label} proposal."
            question = f"When did {speaker} present the {label} proposal?"
            answer = format_date(event_date)
            family_name = "relative-weekday"
        elif family == 6:
            year = 2021 + ((ordinal + sample_index) % 5)
            text = f"I first drafted the {label} plan back in {year}."
            question = f"Which year did {speaker} first draft the {label} plan?"
            answer = str(year)
            family_name = "explicit-year"
        elif family == 7:
            event_date = when - timedelta(days=10 + ((ordinal + sample_index) % 9))
            text = (
                f"I completed the archive review for the {label} on "
                f"{format_date(event_date)}."
            )
            question = (
                f"When did {speaker} complete the archive review for the {label}?"
            )
            answer = format_date(event_date)
            family_name = "explicit-date"
        else:
            anchor_session = builder.session_for(ordinal, 3)
            anchor_date = builder.session_dates[anchor_session]
            chain_length = (
                4
                if sample_index == 0 and ordinal == 8
                else 3
                if (ordinal + sample_index) % 3 == 0
                else 2
            )
            evidence_ids.append(
                builder.add_exchange(
                    anchor_session,
                    speaker,
                    f"Today I attended the kickoff meeting for the {label}.",
                    caption=(
                        f"a kickoff photo from the {label}"
                        if ordinal % 2 == 0
                        else None
                    ),
                    reply=False,
                )
            )
            if chain_length >= 3:
                evidence_ids.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, 4),
                        speaker,
                        (
                            f"The planning checkpoint for the {label} happened "
                            "one day after its kickoff meeting."
                        ),
                        reply=False,
                    )
                )
            if chain_length == 4:
                evidence_ids.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, 5),
                        speaker,
                        (
                            f"The field check for the {label} happened two days "
                            "after its planning checkpoint."
                        ),
                        reply=False,
                    )
                )
            if chain_length == 2:
                review_relation = "two days after its kickoff meeting"
                event_date = anchor_date + timedelta(days=2)
            elif chain_length == 3:
                review_relation = "two days after its planning checkpoint"
                event_date = anchor_date + timedelta(days=3)
            else:
                review_relation = "one day after its field check"
                event_date = anchor_date + timedelta(days=4)
            evidence_ids.append(
                builder.add_exchange(
                    session,
                    speaker,
                    (
                        f"The review for the {label} took place "
                        f"{review_relation}."
                    ),
                    reply=False,
                )
            )
            question = f"When did the review for the {label} take place?"
            answer = format_date(event_date)
            family_name = "cross-session-relative-date-join"
        if not evidence_ids:
            evidence_ids.append(
                builder.add_exchange(
                    session,
                    speaker,
                    text,
                    caption=(
                        f"a contextual photo from the {label}"
                        if ordinal % 2 == 0
                        else None
                    ),
                    reply=False,
                )
            )
        rows.append(
            qa_record(
                sample_index,
                2,
                ordinal,
                question,
                answer,
                evidence_ids,
                family_name,
            )
        )
    return rows


def generate_open_domain(
    builder: ConversationBuilder,
    count: int,
    sample_index: int,
) -> list[dict]:
    rows = []
    countries = list(COUNTRY_BY_CITY.items())
    states = list(STATE_BY_CITY.items())
    for ordinal in range(count):
        speaker = builder.speakers[(ordinal + sample_index) % 2]
        label = label_for(9000 + sample_index * 100 + ordinal)
        family = ordinal % 8
        session = builder.session_for(ordinal, 4)
        evidence = []
        if family == 0:
            city, answer = countries[(ordinal + sample_index) % len(countries)]
            text = f"For the {label}, I spent a week in {city}."
            question = f"Which country did {speaker} visit for the {label}?"
            family_name = "geographic-country-inference"
        elif family == 1:
            city, answer = states[(ordinal + sample_index) % len(states)]
            text = f"During the {label}, I stayed with friends in {city}."
            question = f"Which US state did {speaker} visit during the {label}?"
            family_name = "geographic-state-inference"
        elif family == 2:
            text = f"I loved the long forest hikes and quiet campsites during the {label}."
            question = f"Would {speaker} probably prefer a national park or an indoor arcade after the {label}?"
            answer = "a national park"
            family_name = "preference-inference"
        elif family == 3:
            text = f"For the {label}, I enjoyed teaching children about plants and protecting local habitats."
            question = f"What career might suit {speaker} based on the {label}?"
            answer = "environmental educator"
            family_name = "career-inference"
        elif family == 4:
            text = f"During the {label}, I began running every morning and tracking my distance."
            question = f"What practical gift could support {speaker}'s new habit from the {label}?"
            answer = "a fitness tracker"
            family_name = "commonsense-gift-inference"
        elif family == 5:
            text = f"For the {label}, I kept reading stories about spacecraft, distant planets, and future colonies."
            question = f"What fiction genre would {speaker} likely enjoy after the {label}?"
            answer = "science fiction"
            family_name = "genre-inference"
        elif family == 6:
            text = f"The {label} involved repairing loose screws and assembling wooden shelves."
            question = f"What common tool would {speaker} likely need for the {label}?"
            answer = "a screwdriver"
            family_name = "tool-inference"
        else:
            action_count = 4 + ((ordinal + sample_index) % 14)
            for step in range(action_count):
                evidence.append(
                    builder.add_exchange(
                        builder.session_for(ordinal, step),
                        speaker,
                        (
                            f"During the {label}, I "
                            f"{COMMUNITY_ACTIONS[(ordinal + step) % len(COMMUNITY_ACTIONS)]}."
                        ),
                        caption=(
                            f"a volunteer photo from the {label}"
                            if step == 0 and ordinal % 2 == 0
                            else None
                        ),
                        reply=False,
                    )
                )
            question = (
                f"Based on {speaker}'s actions during the {label}, would {speaker} "
                "likely support another community-service event?"
            )
            answer = "Likely yes"
            family_name = "multi-premise-behavioral-inference"
        if not evidence:
            evidence.append(
                builder.add_exchange(
                    session,
                    speaker,
                    text,
                    caption=(
                        f"a contextual photo from the {label}"
                        if ordinal % 2 == 0
                        else None
                    ),
                )
            )
        if ordinal % 3 == 0 and family != 7:
            evidence.append(
                builder.add_exchange(
                    builder.session_for(ordinal, 5),
                    speaker,
                    f"The {label} remains one of the experiences I talk about positively.",
                )
            )
        rows.append(
            qa_record(
                sample_index,
                3,
                ordinal,
                question,
                answer,
                evidence,
                family_name,
            )
        )
    return rows


def generate_adversarial(
    specs: list[dict],
    count: int,
    sample_index: int,
) -> list[dict]:
    if count > len(specs):
        raise ValueError("Not enough independent single-hop facts for adversarial cases.")
    return [
        qa_record(
            sample_index,
            5,
            ordinal,
            spec["question"],
            None,
            spec["evidence"],
            spec["family"],
        )
        for ordinal, spec in enumerate(specs[:count])
    ]


def generate_sample(
    profile: ConversationProfile,
    sample_index: int,
    seed: int,
) -> dict:
    rng = random.Random(seed + sample_index * 1009)
    builder = ConversationBuilder(profile, rng, sample_index)
    single_rows, adversarial_specs = generate_single_hop(
        builder,
        profile.category_counts[4],
        sample_index,
    )
    multi_rows = generate_multi_hop(
        builder,
        profile.category_counts[1],
        sample_index,
    )
    temporal_rows = generate_temporal(
        builder,
        profile.category_counts[2],
        sample_index,
    )
    open_rows = generate_open_domain(
        builder,
        profile.category_counts[3],
        sample_index,
    )
    adversarial_rows = generate_adversarial(
        adversarial_specs,
        profile.category_counts[5],
        sample_index,
    )
    builder.add_noise()
    return {
        "sample_id": profile.sample_id,
        "conversation": builder.as_conversation(),
        "observation": {},
        "session_summary": {},
        "event_summary": {},
        "qa": multi_rows
        + temporal_rows
        + open_rows
        + single_rows
        + adversarial_rows,
        "synthetic_metadata": {
            "purpose": "development_and_regression_testing_only",
            "official_score": False,
            "profile_index": sample_index,
        },
    }


def generate_dataset(seed: int = DEFAULT_SEED) -> list[dict]:
    return [
        generate_sample(profile, index, seed)
        for index, profile in enumerate(CONVERSATION_PROFILES)
    ]


def normalized_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def validate_dataset(
    dataset: list[dict],
    *,
    official_dataset: list[dict] | None = None,
) -> dict:
    errors = []
    if len(dataset) != len(CONVERSATION_PROFILES):
        errors.append("conversation count differs from the synthetic profile")
    all_questions = []
    all_question_ids = []
    category_counts = Counter()
    family_counts = Counter()
    visual_question_count = 0
    cross_session_multi_count = 0
    turn_counts = []
    for sample_index, (sample, profile) in enumerate(
        zip(dataset, CONVERSATION_PROFILES)
    ):
        if sample.get("sample_id") != profile.sample_id:
            errors.append(f"sample ID mismatch at conversation {sample_index}")
        conversation = sample.get("conversation") or {}
        sessions = {
            key: value
            for key, value in conversation.items()
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
        }
        if len(sessions) != profile.session_count:
            errors.append(f"session count mismatch in {profile.sample_id}")
        if sum(len(turns) for turns in sessions.values()) != profile.target_turn_count:
            errors.append(f"turn count mismatch in {profile.sample_id}")
        evidence_lookup = {}
        session_lookup = {}
        for session_key, turns in sessions.items():
            turn_counts.append(len(turns))
            for turn in turns:
                dia_id = turn.get("dia_id")
                if not dia_id or dia_id in evidence_lookup:
                    errors.append(f"duplicate or missing dialog ID in {profile.sample_id}")
                    continue
                evidence_lookup[dia_id] = turn
                session_lookup[dia_id] = session_key
                if turn.get("speaker") not in {
                    profile.speaker_a,
                    profile.speaker_b,
                }:
                    errors.append(f"unknown speaker in {profile.sample_id}")
                if not str(turn.get("text") or "").strip():
                    errors.append(f"empty turn in {profile.sample_id}")

        sample_counts = Counter()
        for qa in sample.get("qa") or []:
            category = qa.get("category")
            sample_counts[category] += 1
            category_counts[category] += 1
            family_counts[qa.get("synthetic_family")] += 1
            all_questions.append(normalized_text(qa.get("question")))
            all_question_ids.append(qa.get("question_id"))
            evidence = qa.get("evidence") or []
            if not evidence or not set(evidence).issubset(evidence_lookup):
                errors.append(f"invalid evidence in {qa.get('question_id')}")
                continue
            if any("blip_caption" in evidence_lookup[item] for item in evidence):
                visual_question_count += 1
            if category == 1:
                if len(evidence) < 2:
                    errors.append(f"multi-hop evidence too short in {qa.get('question_id')}")
                if len({session_lookup[item] for item in evidence}) < 2:
                    errors.append(f"multi-hop evidence is not cross-session in {qa.get('question_id')}")
                else:
                    cross_session_multi_count += 1
            if category == 4 and len(evidence) != 1:
                errors.append(f"single-hop evidence is not singular in {qa.get('question_id')}")
            if category == 5:
                if qa.get("answer") is not None:
                    errors.append(f"adversarial answer must be null in {qa.get('question_id')}")
            elif qa.get("answer") in {None, ""}:
                errors.append(f"answerable QA has no answer in {qa.get('question_id')}")
        normalized_sample_counts = {
            category: sample_counts.get(category, 0)
            for category in CATEGORY_NAMES
        }
        if normalized_sample_counts != profile.category_counts:
            errors.append(f"category profile mismatch in {profile.sample_id}")

    if dict(sorted(category_counts.items())) != EXPECTED_CATEGORY_COUNTS:
        errors.append("global category distribution differs from LoCoMo profile")
    if len(set(all_question_ids)) != len(all_question_ids):
        errors.append("question IDs are not unique")
    if len(set(all_questions)) != len(all_questions):
        errors.append("questions are not unique")

    official_questions = {
        normalized_text(qa.get("question"))
        for sample in official_dataset or []
        for qa in sample.get("qa") or []
    }
    exact_reuse = set(all_questions).intersection(official_questions)
    if exact_reuse:
        errors.append(f"{len(exact_reuse)} official questions were reused exactly")
    if errors:
        raise ValueError("Synthetic LoCoMo validation failed:\n- " + "\n- ".join(errors))

    conversation_turn_counts = []
    for sample in dataset:
        conversation_turn_counts.append(
            sum(
                len(value)
                for key, value in sample["conversation"].items()
                if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
            )
        )
    return {
        "conversation_count": len(dataset),
        "question_count": sum(category_counts.values()),
        "category_counts": dict(sorted(category_counts.items())),
        "category_names": CATEGORY_NAMES,
        "family_counts": dict(sorted(family_counts.items())),
        "session_count_range": [
            min(profile.session_count for profile in CONVERSATION_PROFILES),
            max(profile.session_count for profile in CONVERSATION_PROFILES),
        ],
        "conversation_turn_count_range": [
            min(conversation_turn_counts),
            max(conversation_turn_counts),
        ],
        "cross_session_multi_hop_count": cross_session_multi_count,
        "visual_evidence_question_count": visual_question_count,
        "exact_official_question_reuse_count": len(exact_reuse),
    }


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_artifacts(
    dataset: list[dict],
    validation: dict,
    *,
    output_path: Path,
    manifest_path: Path,
    seed: int,
    official_data_path: Path | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_content = json_bytes(dataset)
    output_path.write_bytes(dataset_content)
    manifest = {
        "schema_version": 1,
        "dataset_name": "PRAGMOS Synthetic LoCoMo Development Set V1",
        "purpose": "development_and_regression_testing_only",
        "official_locomo_score": False,
        "seed": seed,
        "generator": Path(__file__).name,
        "generator_sha256": sha256_path(Path(__file__)),
        "dataset_file": output_path.name,
        "dataset_sha256": sha256_bytes(dataset_content),
        "official_profile_source": OFFICIAL_REPOSITORY,
        "official_profile_commit": OFFICIAL_COMMIT,
        "official_profile_fields_used": [
            "conversation count",
            "sessions per conversation",
            "turn count per conversation",
            "QA category counts per conversation",
            "public JSON schema",
        ],
        "official_content_fields_not_used_for_generation": [
            "conversation turns",
            "questions",
            "answers",
            "evidence text",
            "speaker names",
            "dates",
            "observations",
            "session summaries",
            "event summaries",
        ],
        "official_question_reuse_check": {
            "performed": official_data_path is not None,
            "official_data_sha256": (
                sha256_path(official_data_path) if official_data_path else None
            ),
            "matched_question_count": validation[
                "exact_official_question_reuse_count"
            ],
        },
        "validation": validation,
        "publication_note": (
            "This synthetic set is for development and regression testing only. "
            "It is not an official LoCoMo split, must not be reported as an "
            "official result, and must be frozen before official evaluation."
        ),
    }
    manifest = json.loads(json.dumps(manifest, ensure_ascii=True))
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmark/locomo/synthetic_v1/pragmos_synthetic_locomo_1986_v1.json",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "benchmark/locomo/synthetic_v1/"
            "pragmos_synthetic_locomo_1986_v1_manifest.json"
        ),
    )
    parser.add_argument(
        "--official-data",
        help="Optional official LoCoMo file used only to reject exact question reuse.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_dataset = None
    official_data_path = None
    if args.official_data:
        official_data_path = Path(args.official_data)
        official_dataset = json.loads(
            official_data_path.read_text(encoding="utf-8")
        )
    dataset = generate_dataset(seed=args.seed)
    validation = validate_dataset(dataset, official_dataset=official_dataset)
    manifest = write_artifacts(
        dataset,
        validation,
        output_path=Path(args.output),
        manifest_path=Path(args.manifest),
        seed=args.seed,
        official_data_path=official_data_path,
    )
    print(f"Wrote dataset: {args.output}")
    print(f"Wrote manifest: {args.manifest}")
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    main()
