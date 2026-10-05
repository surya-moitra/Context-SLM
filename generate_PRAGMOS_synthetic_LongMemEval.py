#!/usr/bin/env python3
"""Generate a reproducible LongMemEval-shaped synthetic development set.

The official dataset is used only as a type/abstention profile. Official
questions, answers, entities, and conversations are never used to construct
synthetic content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_SEED = 20261001
DEFAULT_SESSION_MIN = 38
DEFAULT_SESSION_MAX = 62
QUESTION_TYPES = {
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
}
INSUFFICIENT_ANSWER = "The information provided is not enough."

PEOPLE = [
    "Aarav",
    "Bianca",
    "Caleb",
    "Dalia",
    "Emiko",
    "Farah",
    "Gideon",
    "Hana",
    "Imani",
    "Jonas",
    "Keira",
    "Luca",
    "Maya",
    "Nikhil",
    "Odette",
    "Priya",
    "Rafael",
    "Sofia",
    "Tariq",
    "Yuna",
]
PLACES = [
    "Aster Hall",
    "Beacon Workshop",
    "Cedar House",
    "Driftwood Studio",
    "Elm Street Library",
    "Fern Court",
    "Granite Center",
    "Harbor Annex",
    "Indigo Room",
    "Juniper Pavilion",
    "Kestrel Arts",
    "Lantern Theatre",
    "Maple Commons",
    "Northwind Lab",
    "Orchid Gallery",
    "Pioneer Hall",
]
TITLES = [
    "The Clockmaker's Map",
    "Winter at Marigold Station",
    "The Paper Observatory",
    "Signals Across the Lake",
    "A Garden of Small Moons",
    "The Last Tram Home",
    "Threads of the Monsoon",
    "The Quiet Cartographer",
]
PRODUCTS = [
    "Aurora K2",
    "Boreal Mini",
    "Cobalt Seven",
    "Dune Pro",
    "Ember One",
    "Fjord Lite",
    "Grove X4",
    "Helio Plus",
]
COLORS = [
    "cobalt blue",
    "forest green",
    "marigold yellow",
    "plum purple",
    "coral red",
    "silver gray",
    "teal",
    "ivory",
]


@dataclass(frozen=True)
class SessionSpec:
    timestamp: datetime
    messages: list[dict]
    answer_evidence: bool = False
    tag: str = "noise"


@dataclass(frozen=True)
class Scenario:
    question: str
    answer: str | int | float
    family: str
    sessions: list[SessionSpec]


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y/%m/%d (%a) %H:%M")


def dialogue(
    timestamp: datetime,
    user_text: str,
    assistant_text: str,
    *,
    answer_evidence: bool = False,
    tag: str = "noise",
) -> SessionSpec:
    return SessionSpec(
        timestamp=timestamp,
        messages=[
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        answer_evidence=answer_evidence,
        tag=tag,
    )


def choose_distinct(rng: random.Random, values: list[str], count: int) -> list[str]:
    return rng.sample(values, count)


def single_user_scenario(
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
    abstention: bool,
) -> Scenario:
    when = question_date - timedelta(days=rng.randint(8, 45), hours=2)
    if abstention:
        cases = [
            (
                "How many autographed footballs did I add to my collection?",
                "I added twelve autographed baseballs to my collection this month.",
                "Twelve signed baseballs is a substantial addition.",
                "near-entity-count",
            ),
            (
                "Where did I redeem the coupon for loose-leaf tea?",
                "I redeemed my coupon for coffee beans at Cedar Market.",
                "That coffee coupon sounds useful.",
                "missing-related-place",
            ),
            (
                "What brand is my mirrorless camera body?",
                "My camera bag is made by Northstar, and it has held up well.",
                "Northstar makes sturdy camera bags.",
                "missing-attribute",
            ),
            (
                "On what exact date did I volunteer at the shelter dinner?",
                "I volunteered at the shelter fundraising dinner sometime in March.",
                "That was a generous way to help, though you did not mention the day.",
                "underspecified-date",
            ),
            (
                "Where do I take my Pilates classes?",
                "I take yoga classes at Harbor Annex every Wednesday.",
                "Harbor Annex sounds convenient for yoga.",
                "activity-mismatch",
            ),
            (
                "Which play did my nephew perform in?",
                f"My niece performed in {rng.choice(TITLES)} at school.",
                "That must have been a memorable performance for your niece.",
                "relation-mismatch",
            ),
        ]
        question, user_text, assistant_text, family = cases[ordinal % len(cases)]
        return Scenario(
            question,
            INSUFFICIENT_ANSWER,
            family,
            [
                dialogue(
                    when,
                    user_text,
                    assistant_text,
                    answer_evidence=True,
                    tag="insufficient-evidence",
                )
            ],
        )

    family = ordinal % 8
    if family == 0:
        school = rng.choice(PLACES)
        advisor = rng.choice(PEOPLE)
        degree = rng.choice(
            ["Environmental Economics", "Information Design", "Marine Biology", "Urban Planning"]
        )
        return Scenario(
            f"What degree did I complete through {school} while working with {advisor}?",
            degree,
            "user-degree",
            [
                dialogue(
                    when,
                    f"While working with {advisor} through {school}, I finally completed my degree in {degree}.",
                    f"Congratulations on completing {degree}!",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 1:
        place = rng.choice(PLACES)
        teacher = rng.choice(PEOPLE)
        activity = rng.choice(["pottery", "calligraphy", "salsa", "woodworking"])
        return Scenario(
            f"Where do I take {activity} classes with {teacher}?",
            place,
            "user-place",
            [
                dialogue(
                    when,
                    f"I take my {activity} classes with {teacher} at {place} every Thursday.",
                    f"{place} sounds like a good place to practice {activity}.",
                    answer_evidence=True,
                    tag="answer",
                ),
                dialogue(
                    when - timedelta(days=3),
                    f"I bought new supplies for {activity} near {rng.choice(PLACES)}.",
                    "Having the right supplies should make practice easier.",
                    tag="near-distractor",
                ),
            ],
        )
    if family == 2:
        title = rng.choice(TITLES)
        venue = rng.choice(PLACES)
        companion = rng.choice(PEOPLE)
        return Scenario(
            f"What play did I attend with {companion} at {venue}, the community theatre?",
            title,
            "user-title",
            [
                dialogue(
                    when,
                    f"I attended a production of {title} with {companion} at {venue}, our community theatre.",
                    f"{title} sounds like an interesting production.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 3:
        venue = rng.choice(PLACES)
        organizer = rng.choice(PEOPLE)
        event_date = when.replace(hour=19, minute=0)
        answer = event_date.strftime("%B %-d")
        return Scenario(
            f"When did I attend {organizer}'s neighborhood fundraising dinner at {venue}?",
            answer,
            "user-date",
            [
                dialogue(
                    event_date,
                    f"I attended {organizer}'s neighborhood fundraising dinner at {venue} on {answer}.",
                    "I hope the fundraiser went well.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 4:
        store = rng.choice(PLACES)
        companion = rng.choice(PEOPLE)
        quantity = rng.randint(3, 14)
        item = rng.choice(["ceramic mugs", "mystery novels", "seed packets", "vinyl records"])
        return Scenario(
            f"How many {item} did I buy with {companion} at {store}?",
            quantity,
            "user-quantity",
            [
                dialogue(
                    when,
                    f"I bought exactly {quantity} {item} with {companion} at {store}'s weekend market.",
                    f"That is a nice set of {quantity} {item}.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 5:
        room = rng.choice(PLACES)
        owner = rng.choice(PEOPLE)
        product = rng.choice(PRODUCTS)
        item = rng.choice(["travel kettle", "wireless microphone", "desk projector", "fitness watch"])
        return Scenario(
            f"What model is the {item} that {owner} and I keep in {room}?",
            product,
            "user-model",
            [
                dialogue(
                    when,
                    f"The {item} that {owner} and I keep in {room} is the {product} model.",
                    f"The {product} should be easy to identify later.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 6:
        person = rng.choice(PEOPLE)
        place = rng.choice(PLACES)
        return Scenario(
            f"Who helped me assemble the balcony shelves at {place}?",
            person,
            "user-person",
            [
                dialogue(
                    when,
                    f"{person} helped me assemble the balcony shelves at {place}.",
                    f"It was kind of {person} to help with the shelves.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    duration = rng.choice(["35 minutes", "45 minutes", "one hour", "90 minutes"])
    line = rng.choice(["Cedar Line", "Harbor Line", "Juniper Line", "Orchid Line"])
    destination = rng.choice(PLACES)
    return Scenario(
        f"How long does my commute on the {line} to {destination} take each way?",
        duration,
        "user-duration",
        [
            dialogue(
                when,
                f"My commute on the {line} to {destination} takes {duration} each way on a normal weekday.",
                f"I will remember that the one-way commute is {duration}.",
                answer_evidence=True,
                tag="answer",
            )
        ],
    )


def assistant_scenario(
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
) -> Scenario:
    when = question_date - timedelta(days=rng.randint(8, 45), hours=1)
    family = ordinal % 6
    if family == 0:
        project = rng.choice(PLACES)
        items = [
            "archive researcher",
            "museum caption writer",
            "accessibility tester",
            "community newsletter editor",
            "online language tutor",
            "podcast transcript reviewer",
            "digital pattern designer",
            "genealogy assistant",
            "remote workshop host",
            "catalog quality checker",
        ]
        position = rng.randint(5, 9)
        answer = items[position - 1]
        assistant_text = "Here is the numbered list:\n" + "\n".join(
            f"{index}. {item}" for index, item in enumerate(items, start=1)
        )
        return Scenario(
            f"In the remote-job list for my {project} plan, what was item {position}?",
            answer,
            "assistant-ordinal-list",
            [
                dialogue(
                    when,
                    f"For my {project} plan, please give me ten unusual remote job ideas in a numbered list.",
                    assistant_text,
                    answer_evidence=True,
                    tag="answer",
                ),
                dialogue(
                    when - timedelta(days=4),
                    "Could you list a few office jobs that require commuting?",
                    "1. Records clerk\n2. Reception coordinator\n3. Facilities scheduler",
                    tag="near-distractor",
                ),
            ],
        )
    if family == 1:
        venue = rng.choice(PLACES)
        dish = rng.choice(["saffron noodles", "miso pancakes", "coconut stew", "mango dumplings"])
        traveler = rng.choice(PEOPLE)
        return Scenario(
            f"What was the restaurant you recommended to {traveler} and me for {dish}?",
            venue,
            "assistant-recommendation-name",
            [
                dialogue(
                    when,
                    f"{traveler} and I need one restaurant known for {dish}. What do you recommend?",
                    f"I recommend {venue}; it is especially well known for {dish}.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 2:
        process = rng.choice(
            ["vacuum distillation", "cold fermentation", "solvent extraction", "thermal lamination"]
        )
        facility = rng.choice(["the north plant", "the riverside workshop", "the pilot facility"])
        project = rng.choice(PRODUCTS)
        return Scenario(
            f"For the {project} project, which process did you say is used at {facility}?",
            process,
            "assistant-process",
            [
                dialogue(
                    when,
                    f"For the {project} project, what production process is used at {facility}?",
                    f"For {project}, the process used at {facility} is {process}.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 3:
        phone = f"555-{rng.randint(200, 899)}-{rng.randint(1000, 9999)}"
        office = rng.choice(["visitor bureau", "arts council", "trail information office"])
        region = rng.choice(PLACES)
        return Scenario(
            f"What phone number did you give me for the {region} {office}?",
            phone,
            "assistant-contact-detail",
            [
                dialogue(
                    when,
                    f"Do you have the phone number for the {region} {office}?",
                    f"Yes. The {region} {office} can be reached at {phone}.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    if family == 4:
        color = rng.choice(COLORS)
        creature = rng.choice(["Plesiosaur", "clockwork heron", "moon moth", "river dragon"])
        book = rng.choice(TITLES)
        return Scenario(
            f"In {book}, what color did you say the {creature} was in the illustration?",
            color,
            "assistant-visual-attribute",
            [
                dialogue(
                    when,
                    f"In {book}, describe the {creature} in the illustration.",
                    f"In {book}, the {creature} has a {color} body with small white markings.",
                    answer_evidence=True,
                    tag="answer",
                )
            ],
        )
    objectives = choose_distinct(
        rng,
        [
            "map neighborhood heat islands",
            "compare tree canopy coverage",
            "measure evening surface temperatures",
            "interview residents about shade access",
            "publish an open street-level dataset",
        ],
        3,
    )
    study_name = rng.choice(["River Shade Study", "Cool Streets Study", "Tree Line Study", "Evening Heat Study"])
    study_place = rng.choice(PLACES)
    answer = "; ".join(objectives)
    return Scenario(
        f"What were the three objectives you outlined for the {study_name} at {study_place}?",
        answer,
        "assistant-multi-item",
        [
            dialogue(
                when,
                f"Please outline three objectives for my urban heat project, the {study_name} at {study_place}.",
                "The three objectives are: " + "; ".join(objectives) + ".",
                answer_evidence=True,
                tag="answer",
            )
        ],
    )


def preference_scenario(
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
) -> Scenario:
    when = question_date - timedelta(days=rng.randint(8, 45), hours=3)
    cases = [
        (
            "video editing resources",
            "advanced color grading and keyboard workflows for DaVinci Resolve",
            "generic beginner tutorials or resources for unrelated editors",
        ),
        (
            "a hotel for my coastal trip",
            "a quiet boutique hotel with an ocean view and a rooftop pool",
            "large budget chains without a view",
        ),
        (
            "a weekend cultural event",
            "an event where I can practice Spanish and French through conversation",
            "events without language exchange",
        ),
        (
            "a fitness routine",
            "low-impact swimming and mobility sessions that fit into thirty minutes",
            "high-impact running or workouts longer than an hour",
        ),
        (
            "a restaurant for dinner",
            "a vegetarian Korean restaurant with clearly labeled nut-free dishes",
            "steakhouses or restaurants that cannot handle nut allergies",
        ),
        (
            "a book for my holiday",
            "a character-driven historical mystery under four hundred pages",
            "graphic horror or very long military histories",
        ),
        (
            "a meditation app",
            "an offline app with short unguided timers and no social feed",
            "subscription-only apps centered on group challenges",
        ),
        (
            "a professional conference",
            "a small conference about interpretable machine learning in healthcare",
            "broad marketing conferences unrelated to medicine",
        ),
    ]
    topic, positive, negative = cases[ordinal % len(cases)]
    planning_context = rng.choice(PLACES)
    companion = rng.choice(PEOPLE)
    question = f"For my {planning_context} plans with {companion}, can you recommend {topic}?"
    answer = (
        f"The user would prefer a recommendation for {positive}. "
        f"They would not prefer {negative}."
    )
    return Scenario(
        question,
        answer,
        "preference-constraint-synthesis",
        [
            dialogue(
                when,
                f"For my {planning_context} plans with {companion} and future recommendations, I prefer {positive}. Please avoid {negative}.",
                "Understood. I will use those positive and negative preferences next time.",
                answer_evidence=True,
                tag="answer",
            )
        ],
    )


def knowledge_update_scenario(
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
    abstention: bool,
) -> Scenario:
    times = [
        question_date - timedelta(days=42),
        question_date - timedelta(days=24),
        question_date - timedelta(days=7),
    ]
    if abstention:
        abstention_family = ordinal % 3
        context = rng.choice(PLACES)
        witness = rng.choice(PEOPLE)
        if abstention_family == 1:
            sessions = [
                dialogue(
                    times[0],
                    f"In my {context} notes with {witness}, my work team was the Atlas group.",
                    "The earlier team assignment is noted.",
                    answer_evidence=True,
                    tag="insufficient-update-1",
                ),
                dialogue(
                    times[2],
                    f"I moved from the Atlas group to the Beacon group at work, as I told {witness}.",
                    "The latest team assignment is now the Beacon group.",
                    answer_evidence=True,
                    tag="insufficient-update-2",
                ),
            ]
            return Scenario(
                f"According to my {context} updates with {witness}, what is my current job title?",
                INSUFFICIENT_ANSWER,
                "update-missing-attribute-abstention",
                sessions,
            )
        if abstention_family == 2:
            sessions = [
                dialogue(
                    times[0],
                    f"In my {context} notes, my brother {witness} preferred cats.",
                    "His earlier pet preference is noted.",
                    answer_evidence=True,
                    tag="insufficient-update-1",
                ),
                dialogue(
                    times[2],
                    f"My brother {witness} now prefers dogs instead of cats.",
                    "His latest pet preference is dogs.",
                    answer_evidence=True,
                    tag="insufficient-update-2",
                ),
            ]
            return Scenario(
                f"In my {context} notes, what pet does my sister currently prefer?",
                INSUFFICIENT_ANSWER,
                "update-related-person-abstention",
                sessions,
            )
        old, middle, current = choose_distinct(
            rng,
            ["green tea", "oolong tea", "mint tea", "black tea", "white tea"],
            3,
        )
        sessions = [
            dialogue(
                times[0],
                f"In the {context} notes I shared with {witness}, my preferred afternoon tea was {old}.",
                "I will remember that tea preference.",
                answer_evidence=True,
                tag="insufficient-update-1",
            ),
            dialogue(
                times[1],
                f"For {context}, I told {witness} that I switched my afternoon tea from {old} to {middle}.",
                "I have updated the tea preference.",
                answer_evidence=True,
                tag="insufficient-update-2",
            ),
            dialogue(
                times[2],
                f"My latest {context} update to {witness} says I now prefer {current} as my afternoon tea instead of {middle}.",
                "The latest tea preference is noted.",
                answer_evidence=True,
                tag="insufficient-update-3",
            ),
        ]
        return Scenario(
            f"According to my {context} updates with {witness}, which coffee roast do I currently prefer?",
            INSUFFICIENT_ANSWER,
            "update-near-entity-abstention",
            sessions,
        )

    families = [
        ("favorite weekend animal", ["cats", "dogs", "parrots"]),
        ("preferred morning drink", ["filter coffee", "masala chai", "cocoa"]),
        ("home city", ["Pune", "Mysuru", "Kochi"]),
        ("work laptop", ["Aurora K2", "Fjord Lite", "Grove X4"]),
        ("job title", ["Data Analyst", "Analytics Lead", "Research Manager"]),
        ("favorite running route", ["Canal Loop", "Garden Trail", "Harbor Path"]),
        ("music subscription", ["TuneBox", "WaveRoom", "EchoShelf"]),
        ("weekly class", ["pottery", "salsa", "watercolor"]),
    ]
    attribute, values = families[ordinal % len(families)]
    context = rng.choice(PLACES)
    witness = rng.choice(PEOPLE)
    old, middle, current = values
    sessions = [
        dialogue(
            times[0],
            f"In the {context} notes I shared with {witness}, my {attribute} used to be {old}.",
            f"I will remember that the earlier value was {old}.",
            answer_evidence=True,
            tag="update-old",
        ),
        dialogue(
            times[1],
            f"For {context}, I told {witness} that I changed my {attribute} from {old} to {middle}.",
            f"I have updated it to {middle}.",
            answer_evidence=True,
            tag="update-middle",
        ),
        dialogue(
            times[2],
            f"My latest {context} update to {witness} changes it again: my {attribute} is now {current}, not {middle}.",
            f"The current value is now {current}.",
            answer_evidence=True,
            tag="update-current",
        ),
    ]
    mode = (ordinal // len(families)) % 3
    if mode == 0:
        question = f"According to my {context} updates with {witness}, what is my current {attribute}?"
        answer = current
        family = "update-current"
    elif mode == 1:
        question = f"In my {context} updates with {witness}, what was my {attribute} immediately before {current}?"
        answer = middle
        family = "update-previous"
    else:
        question = f"For my {context} notes with {witness}, what was my original {attribute}, before either update?"
        answer = old
        family = "update-original"
    return Scenario(question, answer, family, sessions)


def temporal_scenario(
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
    abstention: bool,
) -> Scenario:
    if abstention:
        abstention_family = ordinal % 3
        first_date = question_date - timedelta(days=18)
        workshop_place = rng.choice(PLACES)
        tour_place = rng.choice([place for place in PLACES if place != workshop_place])
        if abstention_family == 1:
            return Scenario(
                f"Which happened first at {workshop_place}: my print workshop or my sculpture class?",
                INSUFFICIENT_ANSWER,
                "temporal-undated-events",
                [
                    dialogue(
                        question_date - timedelta(days=20),
                        f"I remember enjoying both the print workshop and sculpture class at {workshop_place}, but I did not record their dates.",
                        "Without dates or an explicit order, their sequence is unknown.",
                        answer_evidence=True,
                        tag="undated-events",
                    )
                ],
            )
        if abstention_family == 2:
            return Scenario(
                f"How many weeks ago did I finish the mosaic at {workshop_place}?",
                INSUFFICIENT_ANSWER,
                "temporal-vague-date",
                [
                    dialogue(
                        question_date - timedelta(days=9),
                        f"I said I had finished the mosaic at {workshop_place} recently, but I did not say when.",
                        "The event is known, but there is no exact date for a week calculation.",
                        answer_evidence=True,
                        tag="vague-date",
                    )
                ],
            )
        return Scenario(
            f"How many days passed between my astronomy workshop at {workshop_place} and my harbor tour from {tour_place}?",
            INSUFFICIENT_ANSWER,
            "temporal-missing-endpoint",
            [
                dialogue(
                    first_date,
                    f"I attended the astronomy workshop at {workshop_place} today.",
                    "I hope the workshop was illuminating.",
                    answer_evidence=True,
                    tag="known-event-date",
                ),
                dialogue(
                    question_date - timedelta(days=5),
                    f"I am considering a harbor tour from {tour_place}, but I have not booked a date.",
                    "You will need a confirmed date before comparing the two events.",
                    answer_evidence=True,
                    tag="missing-event-date",
                ),
            ],
        )

    family = ordinal % 6
    if family == 0:
        gap = rng.randint(4, 21)
        first = question_date - timedelta(days=gap + 12)
        second = first + timedelta(days=gap)
        event_a = rng.choice(["ceramics fair", "botanical lecture", "night market"])
        event_b = rng.choice(["railway museum visit", "poetry reading", "harbor tour"])
        place_a, place_b = choose_distinct(rng, PLACES, 2)
        return Scenario(
            f"How many days passed between the {event_a} at {place_a} and the {event_b} at {place_b}?",
            f"{gap} days",
            "temporal-date-difference",
            [
                dialogue(
                    first,
                    f"I went to the {event_a} at {place_a} today.",
                    "That sounds like a good outing.",
                    answer_evidence=True,
                    tag="event-a",
                ),
                dialogue(
                    second,
                    f"I visited the {event_b} at {place_b} today.",
                    "I hope you enjoyed the visit.",
                    answer_evidence=True,
                    tag="event-b",
                ),
            ],
        )
    if family == 1:
        weeks = rng.randint(2, 7)
        event_question, event_statement = rng.choice(
            [
                ("receive the brass lamp", "received the brass lamp"),
                ("meet up with my aunt", "met up with my aunt"),
                ("finish the mosaic", "finished the mosaic"),
            ]
        )
        place = rng.choice(PLACES)
        event_date = question_date - timedelta(weeks=weeks)
        return Scenario(
            f"How many weeks ago did I {event_question} at {place}?",
            weeks,
            "temporal-relative-weeks",
            [
                dialogue(
                    event_date,
                    f"Today I {event_statement} at {place}.",
                    "I will keep that event in context.",
                    answer_evidence=True,
                    tag="dated-event",
                )
            ],
        )
    if family == 2:
        events = choose_distinct(
            rng,
            [
                "painted the hallway",
                "ordered the custom notebook",
                "visited the orchid show",
                "helped prepare the nursery",
                "repaired the garden gate",
            ],
            3,
        )
        dates = [question_date - timedelta(days=30), question_date - timedelta(days=19), question_date - timedelta(days=8)]
        sessions = [
            dialogue(
                date,
                f"Today I {event}.",
                "That event is noted.",
                answer_evidence=True,
                tag=f"ordered-event-{index}",
            )
            for index, (event, date) in enumerate(zip(events, dates), start=1)
        ]
        return Scenario(
            "In what order did these happen: " + ", ".join(events) + "?",
            "First, " + events[0] + "; then, " + events[1] + "; finally, " + events[2] + ".",
            "temporal-event-order",
            sessions,
        )
    if family == 3:
        before = question_date - timedelta(days=17)
        after = before + timedelta(days=1)
        event_a = rng.choice(["attended the chess meetup", "bought the blue suitcase", "completed the first-aid course"])
        event_b = rng.choice(["started the bread workshop", "visited the sculpture garden", "joined the choir rehearsal"])
        place_a, place_b = choose_distinct(rng, PLACES, 2)
        return Scenario(
            f"What did I do the day after I {event_a} at {place_a}?",
            event_b,
            "temporal-adjacent-day",
            [
                dialogue(
                    before,
                    f"Today I {event_a} at {place_a}.",
                    "That is noted.",
                    answer_evidence=True,
                    tag="reference-day",
                ),
                dialogue(
                    after,
                    f"Today I {event_b} at {place_b}.",
                    "That is noted as well.",
                    answer_evidence=True,
                    tag="joined-day",
                ),
            ],
        )
    if family == 4:
        start_hour = rng.randint(8, 13)
        duration = rng.choice([2, 3, 4, 5])
        date = question_date - timedelta(days=11)
        workshop = rng.choice(["bookbinding workshop", "mapmaking workshop", "repair workshop", "print workshop"])
        place = rng.choice(PLACES)
        return Scenario(
            f"How long did the {workshop} at {place} last?",
            f"{duration} hours",
            "temporal-clock-duration",
            [
                dialogue(
                    date,
                    f"The {workshop} at {place} started at {start_hour}:00 and ended at {start_hour + duration}:00 today.",
                    "That gives the workshop clear start and end times.",
                    answer_evidence=True,
                    tag="timed-event",
                )
            ],
        )
    event_date = question_date - timedelta(days=rng.randint(7, 35))
    event = rng.choice(["lantern festival", "river cleanup", "printmaking class"])
    place = rng.choice(PLACES)
    return Scenario(
        f"On what date did I attend the {event} at {place}?",
        event_date.strftime("%B %-d"),
        "temporal-exact-date",
        [
            dialogue(
                event_date,
                f"I attended the {event} at {place} today.",
                "I hope it was worthwhile.",
                answer_evidence=True,
                tag="dated-event",
            )
        ],
    )


def multi_session_scenario(
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
    abstention: bool,
) -> Scenario:
    base = question_date - timedelta(days=45)
    if abstention:
        abstention_family = ordinal % 4
        price = rng.randint(80, 240)
        trip = rng.choice(PLACES)
        companion = rng.choice(PEOPLE)
        if abstention_family == 1:
            return Scenario(
                f"For my {trip} collection with {companion}, how many autographed footballs did I acquire?",
                INSUFFICIENT_ANSWER,
                "multi-near-entity-abstention",
                [
                    dialogue(
                        base + timedelta(days=8),
                        f"For the {trip} collection, I acquired six autographed baseballs with {companion}.",
                        "The six signed baseballs are noted.",
                        answer_evidence=True,
                        tag="wrong-category-1",
                    ),
                    dialogue(
                        base + timedelta(days=21),
                        f"I added four more autographed baseballs to the {trip} collection.",
                        "That brings the recorded baseball additions to ten.",
                        answer_evidence=True,
                        tag="wrong-category-2",
                    ),
                ],
            )
        if abstention_family == 2:
            known = rng.randint(25, 70)
            return Scenario(
                f"For my {trip} workspace with {companion}, what was the total spent on the lamp and monitor stand?",
                INSUFFICIENT_ANSWER,
                "multi-planned-not-completed-abstention",
                [
                    dialogue(
                        base + timedelta(days=7),
                        f"I bought the workspace lamp for ${known}.",
                        "The completed lamp purchase is noted.",
                        answer_evidence=True,
                        tag="completed-purchase",
                    ),
                    dialogue(
                        base + timedelta(days=24),
                        "I might buy a monitor stand next month, but I have neither chosen nor purchased one.",
                        "There is no completed monitor-stand cost yet.",
                        answer_evidence=True,
                        tag="planned-purchase",
                    ),
                ],
            )
        if abstention_family == 3:
            first = rng.randint(25, 55)
            return Scenario(
                f"With {companion}, what was the average duration of my three practice sessions at {trip}?",
                INSUFFICIENT_ANSWER,
                "multi-missing-measurement-abstention",
                [
                    dialogue(
                        base + timedelta(days=6),
                        f"My first practice session at {trip} lasted {first} minutes.",
                        "The first duration is recorded.",
                        answer_evidence=True,
                        tag="known-duration",
                    ),
                    dialogue(
                        base + timedelta(days=18),
                        f"I completed the second and third practice sessions at {trip}, but I forgot to record their durations.",
                        "The sessions occurred, but their durations are missing.",
                        answer_evidence=True,
                        tag="missing-durations",
                    ),
                ],
            )
        return Scenario(
            f"For my {trip} trip with {companion}, what was the total cost of my new headphones and tablet?",
            INSUFFICIENT_ANSWER,
            "multi-missing-operand",
            [
                dialogue(
                    base + timedelta(days=8),
                    f"For my {trip} trip with {companion}, I bought new headphones for ${price}.",
                    "I hope the headphones sound good.",
                    answer_evidence=True,
                    tag="known-operand",
                ),
                dialogue(
                    base + timedelta(days=20),
                    f"For the {trip} trip with {companion}, I compared several tablets but did not buy one.",
                    "A comparison is useful, but there is no tablet purchase price yet.",
                    answer_evidence=True,
                    tag="missing-operand",
                ),
            ],
        )

    family = ordinal % 10
    if family == 0:
        trip = rng.choice(PLACES)
        companion = rng.choice(PEOPLE)
        items = choose_distinct(
            rng,
            ["linen jacket", "walking boots", "wool scarf", "raincoat", "canvas trousers"],
            3,
        )
        sessions = [
            dialogue(
                base + timedelta(days=5 + index * 7),
                f"For my {trip} trip with {companion}, I need to pick up my {item} from the store.",
                f"I will remember the pending {item} pickup.",
                answer_evidence=True,
                tag=f"count-item-{index}",
            )
            for index, item in enumerate(items)
        ]
        return Scenario(
            f"How many clothing items do I need to pick up for my {trip} trip with {companion}?",
            len(items),
            "multi-count",
            sessions,
        )
    if family == 1:
        show = rng.choice(PLACES)
        teammate = rng.choice(PEOPLE)
        models = choose_distinct(
            rng,
            ["Falcon glider", "Orion capsule", "Tiger tank", "Harbor tug", "Comet racer"],
            4,
        )
        sessions = []
        for index, model in enumerate(models + [models[1]]):
            sessions.append(
                dialogue(
                    base + timedelta(days=4 + index * 6),
                    f"With {teammate} for the model show at {show}, I worked on my {model} kit today.",
                    "That model-building session is noted.",
                    answer_evidence=True,
                    tag=f"distinct-model-{index}",
                )
            )
        return Scenario(
            f"How many distinct model kits have I worked on with {teammate} for the show at {show}?",
            len(models),
            "multi-count-distinct",
            sessions,
        )
    if family == 2:
        workspace = rng.choice(PLACES)
        colleague = rng.choice(PEOPLE)
        prices = [rng.randint(20, 90) for _ in range(3)]
        items = ["desk lamp", "keyboard stand", "webcam cover"]
        sessions = [
            dialogue(
                base + timedelta(days=7 + index * 8),
                f"For my workspace with {colleague} at {workspace}, I bought the {item} for ${price}.",
                "That completed purchase is noted.",
                answer_evidence=True,
                tag=f"money-{index}",
            )
            for index, (item, price) in enumerate(zip(items, prices))
        ]
        return Scenario(
            f"For my workspace with {colleague} at {workspace}, what total did I spend on the desk lamp, keyboard stand, and webcam cover?",
            f"${sum(prices)}",
            "multi-sum-money",
            sessions,
        )
    if family == 3:
        plan = rng.choice(PLACES)
        coach = rng.choice(PEOPLE)
        minutes = [rng.choice([30, 45, 60, 75]), rng.choice([40, 50, 70, 80]), rng.choice([25, 35, 55, 65])]
        activities = ["swimming", "cycling", "walking"]
        sessions = [
            dialogue(
                base + timedelta(days=6 + index * 9),
                f"For my {plan} training plan with {coach}, I spent {value} minutes {activity} today.",
                "That exercise duration is noted.",
                answer_evidence=True,
                tag=f"duration-{index}",
            )
            for index, (activity, value) in enumerate(zip(activities, minutes))
        ]
        return Scenario(
            f"For my {plan} plan with {coach}, how many minutes did I spend across swimming, cycling, and walking?",
            f"{sum(minutes)} minutes",
            "multi-sum-duration",
            sessions,
        )
    if family == 4:
        challenge = rng.choice(TITLES)
        partner = rng.choice(PEOPLE)
        values = [rng.randint(20, 60) for _ in range(3)]
        sessions = [
            dialogue(
                base + timedelta(days=5 + index * 10),
                f"For my {challenge} reading challenge with {partner}, today's session lasted {value} minutes.",
                "That reading duration is noted.",
                answer_evidence=True,
                tag=f"average-{index}",
            )
            for index, value in enumerate(values)
        ]
        average = sum(values) / len(values)
        answer = f"{average:g} minutes"
        return Scenario(
            f"What was the average duration of my three {challenge} reading sessions with {partner}?",
            answer,
            "multi-average",
            sessions,
        )
    if family == 5:
        destination = rng.choice(PLACES)
        colleague = rng.choice(PEOPLE)
        first = rng.randint(25, 55)
        second = first + rng.randint(8, 25)
        sessions = [
            dialogue(
                base + timedelta(days=9),
                f"My bus commute with {colleague} to {destination} took {first} minutes today.",
                "The bus time is noted.",
                answer_evidence=True,
                tag="difference-a",
            ),
            dialogue(
                base + timedelta(days=24),
                f"My train commute with {colleague} to {destination} took {second} minutes today.",
                "The train time is noted.",
                answer_evidence=True,
                tag="difference-b",
            ),
        ]
        return Scenario(
            f"With {colleague}, what is the difference between my train and bus commute times to {destination}?",
            f"{second - first} minutes",
            "multi-difference",
            sessions,
        )
    if family == 6:
        venues = choose_distinct(rng, PLACES, 3)
        visits = [rng.randint(2, 5), rng.randint(6, 9), rng.randint(10, 14)]
        paired = list(zip(venues, visits))
        rng.shuffle(paired)
        sessions = [
            dialogue(
                base + timedelta(days=5 + index * 9),
                f"I have visited {venue} {count} times this year.",
                "That visit count is noted.",
                answer_evidence=True,
                tag=f"argmax-{index}",
            )
            for index, (venue, count) in enumerate(paired)
        ]
        answer = max(paired, key=lambda item: item[1])[0]
        return Scenario(
            "Which of these venues have I visited most this year: "
            + ", ".join(venues)
            + "?",
            answer,
            "multi-argmax",
            sessions,
        )
    if family == 7:
        tour = rng.choice(PLACES)
        companion = rng.choice(PEOPLE)
        distances = [rng.randint(3, 12), rng.randint(5, 15), rng.randint(4, 13)]
        routes = ["canal route", "park route", "harbor route"]
        sessions = [
            dialogue(
                base + timedelta(days=8 + index * 8),
                f"For my cycling tour with {companion} based at {tour}, I covered {distance} kilometers on the {route}.",
                "That route distance is noted.",
                answer_evidence=True,
                tag=f"distance-{index}",
            )
            for index, (route, distance) in enumerate(zip(routes, distances))
        ]
        return Scenario(
            f"For my {tour} cycling tour with {companion}, what total distance did I cover on the canal, park, and harbor routes?",
            f"{sum(distances)} kilometers",
            "multi-total-distance",
            sessions,
        )
    if family == 8:
        project = rng.choice(PRODUCTS)
        completed = rng.randint(6, 15)
        deferred = rng.randint(2, 5)
        sessions = [
            dialogue(
                base + timedelta(days=10),
                f"For project {project}, I completed {completed} review tickets.",
                "The completed-ticket count is noted.",
                answer_evidence=True,
                tag="ratio-completed",
            ),
            dialogue(
                base + timedelta(days=26),
                f"For project {project}, I deferred {deferred} review tickets.",
                "The deferred-ticket count is noted.",
                answer_evidence=True,
                tag="ratio-deferred",
            ),
        ]
        return Scenario(
            f"For project {project}, what was the ratio of completed to deferred review tickets?",
            f"{completed}:{deferred}",
            "multi-ratio",
            sessions,
        )
    venues = choose_distinct(rng, PLACES, 3)
    prices = [rng.randint(18, 35), rng.randint(36, 55), rng.randint(56, 80)]
    paired = list(zip(venues, prices))
    rng.shuffle(paired)
    sessions = [
        dialogue(
            base + timedelta(days=6 + index * 10),
            f"A day pass at {venue} cost me ${price}.",
            "That completed day-pass cost is noted.",
            answer_evidence=True,
            tag=f"argmin-{index}",
        )
        for index, (venue, price) in enumerate(paired)
    ]
    return Scenario(
        "Which venue had the least expensive day pass: " + ", ".join(venues) + "?",
        min(paired, key=lambda item: item[1])[0],
        "multi-argmin",
        sessions,
    )
NOISE_TOPICS = [
    ("balcony herbs", "I moved the basil into a larger clay pot and kept the mint in shade.", "Separating their light needs should help both herbs grow."),
    ("bread recipe", "I bookmarked a rye bread recipe that uses caraway and a long overnight rise.", "An overnight rise can build flavor without requiring constant attention."),
    ("desk lighting", "I changed my desk lamp to warm amber light for evening reading.", "Warmer light may feel more comfortable late in the day."),
    ("library books", "I returned two library books and renewed a collection of travel essays.", "Renewing the essays gives you more time without another trip."),
    ("bicycle care", "I cleaned the bicycle chain and checked the tire pressure before the weekend.", "Regular chain cleaning and tire checks can prevent avoidable problems."),
    ("meal planning", "I planned lentil soup, roasted vegetables, and rice for weekday lunches.", "Those dishes should store and reheat well for several lunches."),
    ("music practice", "I practiced major scales slowly with a metronome before learning a new melody.", "Slow practice can make the later melody work more accurate."),
    ("photo archive", "I sorted old photographs into folders by year and added brief location labels.", "Year and location labels should make the archive easier to search."),
    ("home repair", "I tightened the loose cabinet hinge and replaced one worn felt pad.", "That should stop the door from shifting and protect the floor."),
    ("language study", "I reviewed irregular verbs and recorded short pronunciation exercises.", "Recording yourself is a practical way to notice pronunciation changes."),
    ("garden birds", "I saw two bulbuls near the feeder and refilled the shallow water dish.", "Fresh water can be especially useful to visiting birds."),
    ("budget notes", "I grouped this month's receipts into groceries, transport, and household supplies.", "Those categories should make the monthly comparison clearer."),
    ("museum plan", "I saved the opening hours for a small textile museum for a future visit.", "Checking the hours early will make the visit easier to schedule."),
    ("coding notes", "I wrote a small test for the CSV import and documented the expected column names.", "The test and schema note should reduce import mistakes later."),
    ("weather kit", "I put a compact umbrella and a light rain shell into my day bag.", "That is a sensible kit for changeable weather."),
    ("coffee method", "I adjusted the coffee grind slightly coarser and shortened the brewing time.", "Changing one variable at a time will make the result easier to judge."),
]


def noise_session(
    rng: random.Random,
    timestamp: datetime,
    record_index: int,
    noise_index: int,
) -> SessionSpec:
    topic, user_core, assistant_core = rng.choice(NOISE_TOPICS)
    person = rng.choice(PEOPLE)
    place = rng.choice(PLACES)
    user_text = (
        f"A small unrelated update about {topic}: {user_core} "
        f"I also told {person} that I might stop near {place} later in the week. "
        "This is just a planning note, not a request to change any earlier preference."
    )
    assistant_text = (
        f"{assistant_core} I will treat the note about {person} and {place} as tentative. "
        "The practical next step is to keep the plan simple and verify details when they become definite."
    )
    return dialogue(
        timestamp,
        user_text,
        assistant_text,
        tag=f"noise-{record_index}-{noise_index}",
    )


def build_scenario(
    question_type: str,
    rng: random.Random,
    ordinal: int,
    question_date: datetime,
    abstention: bool,
) -> Scenario:
    if question_type == "single-session-user":
        return single_user_scenario(rng, ordinal, question_date, abstention)
    if question_type == "single-session-assistant":
        if abstention:
            raise ValueError("Synthetic assistant profile unexpectedly requested abstention.")
        return assistant_scenario(rng, ordinal, question_date)
    if question_type == "single-session-preference":
        if abstention:
            raise ValueError("Synthetic preference profile unexpectedly requested abstention.")
        return preference_scenario(rng, ordinal, question_date)
    if question_type == "knowledge-update":
        return knowledge_update_scenario(rng, ordinal, question_date, abstention)
    if question_type == "temporal-reasoning":
        return temporal_scenario(rng, ordinal, question_date, abstention)
    if question_type == "multi-session":
        return multi_session_scenario(rng, ordinal, question_date, abstention)
    raise ValueError(f"Unsupported question type: {question_type}")


def profile_from_official(records: list[dict]) -> list[dict]:
    profile = []
    for index, record in enumerate(records):
        question_type = record.get("question_type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"Unsupported question type at index {index}: {question_type!r}")
        question_id = str(record.get("question_id") or "")
        profile.append(
            {
                "source_index": index,
                "question_type": question_type,
                "abstention": "_abs" in question_id,
            }
        )
    return profile


def record_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def build_record(
    profile_row: dict,
    index: int,
    type_ordinal: int,
    seed: int,
    session_min: int,
    session_max: int,
) -> tuple[dict, str]:
    rng = random.Random(record_seed(seed, index))
    question_type = profile_row["question_type"]
    abstention = bool(profile_row["abstention"])
    question_date = datetime(2025, 1, 15, 20, 0) + timedelta(
        days=index * 3,
        minutes=rng.randint(0, 480),
    )
    scenario = build_scenario(
        question_type,
        rng,
        type_ordinal,
        question_date,
        abstention,
    )
    target_session_count = rng.randint(session_min, session_max)
    sessions = list(scenario.sessions)
    noise_needed = target_session_count - len(sessions)
    if noise_needed < 0:
        raise ValueError("Session range is too small for generated evidence.")
    for noise_index in range(noise_needed):
        days_back = rng.randint(1, 60)
        minute_offset = rng.randint(0, 1439)
        timestamp = question_date - timedelta(days=days_back, minutes=minute_offset)
        sessions.append(noise_session(rng, timestamp, index, noise_index))

    sessions.sort(key=lambda item: (item.timestamp, item.tag))
    suffix = "_abs" if abstention else ""
    question_id = f"synthetic_v1_{index:04d}{suffix}"
    session_ids = []
    session_dates = []
    haystack_sessions = []
    answer_session_ids = []
    answer_number = 0
    noise_number = 0
    for session in sessions:
        if session.answer_evidence:
            answer_number += 1
            session_id = f"syn_{index:04d}_answer{suffix}_{answer_number}"
            answer_session_ids.append(session_id)
        else:
            noise_number += 1
            session_id = f"syn_{index:04d}_noise_{noise_number}"
        session_ids.append(session_id)
        session_dates.append(format_timestamp(session.timestamp))
        haystack_sessions.append(session.messages)

    return (
        {
            "question_id": question_id,
            "question_type": question_type,
            "question": scenario.question,
            "answer": scenario.answer,
            "question_date": format_timestamp(question_date),
            "answer_session_ids": answer_session_ids,
            "haystack_session_ids": session_ids,
            "haystack_dates": session_dates,
            "haystack_sessions": haystack_sessions,
        },
        scenario.family,
    )


def disambiguate_duplicate_question(record: dict, index: int) -> None:
    """Add a supported natural context only when two question texts collide."""
    place = PLACES[index % len(PLACES)]
    person = PEOPLE[(index // len(PLACES)) % len(PEOPLE)]
    project = PRODUCTS[
        (index // (len(PLACES) * len(PEOPLE))) % len(PRODUCTS)
    ]
    context = f"{project} planning notes from {place} reviewed with {person}"
    question = str(record["question"]).strip()
    record["question"] = f"In my {context}, {question[0].lower()}{question[1:]}"

    answer_ids = set(record.get("answer_session_ids") or [])
    for session_id, session in zip(
        record.get("haystack_session_ids") or [],
        record.get("haystack_sessions") or [],
    ):
        if session_id not in answer_ids:
            continue
        for message in session:
            if message.get("role") == "user":
                message["content"] = (
                    f"In my {context}, {message.get('content', '')}"
                )
                break


def generate_records(
    profile: list[dict],
    *,
    seed: int = DEFAULT_SEED,
    session_min: int = DEFAULT_SESSION_MIN,
    session_max: int = DEFAULT_SESSION_MAX,
) -> tuple[list[dict], Counter]:
    if session_min <= 0 or session_max < session_min:
        raise ValueError("Invalid session range.")
    records = []
    family_counts = Counter()
    type_ordinals = Counter()
    seen_questions = set()
    for index, profile_row in enumerate(profile):
        question_type = profile_row["question_type"]
        record, family = build_record(
            profile_row,
            index,
            type_ordinals[question_type],
            seed,
            session_min,
            session_max,
        )
        question_key = normalized_text(record["question"])
        if question_key in seen_questions:
            disambiguate_duplicate_question(record, index)
            question_key = normalized_text(record["question"])
        if question_key in seen_questions:
            raise ValueError(f"Could not make synthetic question unique at index {index}.")
        seen_questions.add(question_key)
        type_ordinals[question_type] += 1
        family_counts[family] += 1
        records.append(record)
    return records, family_counts


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def validate_records(
    records: list[dict],
    profile: list[dict],
    *,
    official_records: list[dict] | None = None,
    session_min: int = DEFAULT_SESSION_MIN,
    session_max: int = DEFAULT_SESSION_MAX,
) -> dict:
    errors = []
    if len(records) != len(profile):
        errors.append(f"record count {len(records)} does not match profile {len(profile)}")

    ids = [str(record.get("question_id")) for record in records]
    if len(set(ids)) != len(ids):
        errors.append("question IDs are not unique")
    questions = [normalized_text(record.get("question")) for record in records]
    if len(set(questions)) != len(questions):
        errors.append("synthetic questions are not unique")

    official_questions = {
        normalized_text(record.get("question")) for record in official_records or []
    }
    reused_questions = sorted(set(questions).intersection(official_questions))
    if reused_questions:
        errors.append(f"{len(reused_questions)} official questions were reused exactly")

    for index, (record, profile_row) in enumerate(zip(records, profile)):
        if record.get("question_type") != profile_row.get("question_type"):
            errors.append(f"type mismatch at index {index}")
        expected_abs = bool(profile_row.get("abstention"))
        actual_abs = "_abs" in str(record.get("question_id"))
        if expected_abs != actual_abs:
            errors.append(f"abstention mismatch at index {index}")
        if expected_abs and record.get("answer") != INSUFFICIENT_ANSWER:
            errors.append(f"abstention answer mismatch at index {index}")
        if not expected_abs and record.get("answer") == INSUFFICIENT_ANSWER:
            errors.append(f"answerable record marked insufficient at index {index}")

        sessions = record.get("haystack_sessions") or []
        session_ids = record.get("haystack_session_ids") or []
        dates = record.get("haystack_dates") or []
        if not (len(sessions) == len(session_ids) == len(dates)):
            errors.append(f"parallel session arrays differ at index {index}")
            continue
        if not session_min <= len(sessions) <= session_max:
            errors.append(f"session count out of range at index {index}: {len(sessions)}")
        if len(set(session_ids)) != len(session_ids):
            errors.append(f"duplicate session IDs at index {index}")
        answer_ids = record.get("answer_session_ids") or []
        if not answer_ids or not set(answer_ids).issubset(session_ids):
            errors.append(f"invalid answer session IDs at index {index}")
        try:
            question_timestamp = datetime.strptime(
                record.get("question_date", ""),
                "%Y/%m/%d (%a) %H:%M",
            )
            session_timestamps = [
                datetime.strptime(value, "%Y/%m/%d (%a) %H:%M")
                for value in dates
            ]
            if session_timestamps != sorted(session_timestamps):
                errors.append(f"session dates are not chronological at index {index}")
            if any(value >= question_timestamp for value in session_timestamps):
                errors.append(f"session date is not before question date at index {index}")
        except (TypeError, ValueError):
            errors.append(f"invalid timestamp format at index {index}")
        for session in sessions:
            if not isinstance(session, list) or not session:
                errors.append(f"empty session at index {index}")
                break
            for message in session:
                if message.get("role") not in {"user", "assistant"}:
                    errors.append(f"invalid role at index {index}")
                    break
                if not str(message.get("content") or "").strip():
                    errors.append(f"empty message at index {index}")
                    break

    profile_types = Counter(row["question_type"] for row in profile)
    record_types = Counter(record["question_type"] for record in records)
    profile_abs = Counter(
        row["question_type"] for row in profile if row.get("abstention")
    )
    record_abs = Counter(
        record["question_type"]
        for record in records
        if "_abs" in record["question_id"]
    )
    if profile_types != record_types:
        errors.append("question-type distribution differs from profile")
    if profile_abs != record_abs:
        errors.append("abstention distribution differs from profile")
    if errors:
        raise ValueError("Synthetic dataset validation failed:\n- " + "\n- ".join(errors))

    session_counts = [len(record["haystack_sessions"]) for record in records]
    return {
        "record_count": len(records),
        "question_type_counts": dict(sorted(record_types.items())),
        "abstention_counts": dict(sorted(record_abs.items())),
        "abstention_total": sum(record_abs.values()),
        "session_count": {
            "min": min(session_counts) if session_counts else 0,
            "max": max(session_counts) if session_counts else 0,
            "mean": (sum(session_counts) / len(session_counts)) if session_counts else 0.0,
            "total": sum(session_counts),
        },
        "exact_official_question_reuse_count": len(reused_questions),
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def write_artifacts(
    records: list[dict],
    family_counts: Counter,
    validation: dict,
    *,
    output_path: Path,
    manifest_path: Path,
    profile_path: Path,
    seed: int,
    session_min: int,
    session_max: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_bytes = json_bytes(records)
    output_path.write_bytes(dataset_bytes)
    manifest = {
        "schema_version": 1,
        "dataset_name": "PRAGMOS Synthetic LongMemEval Development Set V1",
        "purpose": "development_and_regression_testing_only",
        "official_longmemeval_score": False,
        "seed": seed,
        "generator": Path(__file__).name,
        "generator_sha256": sha256_path(Path(__file__)),
        "dataset_file": output_path.name,
        "dataset_sha256": sha256_bytes(dataset_bytes),
        "profile_source_file": str(profile_path),
        "profile_source_sha256": sha256_path(profile_path),
        "profile_fields_used": ["question_type", "question_id contains _abs"],
        "content_fields_not_used_for_generation": [
            "question",
            "answer",
            "answer_session_ids",
            "haystack_dates",
            "haystack_session_ids",
            "haystack_sessions",
            "question_date",
        ],
        "session_range": [session_min, session_max],
        "validation": validation,
        "family_counts": dict(sorted(family_counts.items())),
        "publication_note": (
            "This synthetic set may be used for development and regression tests. "
            "It must not be reported as an official LongMemEval result or as an "
            "independent held-out benchmark."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(json_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-data",
        default="benchmark/longMemEval/longmemeval_s_cleaned.json",
        help="Official-format data used only for type and abstention profile.",
    )
    parser.add_argument(
        "--output",
        default=(
            "benchmark/longMemEval/synthetic_v1/"
            "pragmos_synthetic_longmemeval_500_v1.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "benchmark/longMemEval/synthetic_v1/"
            "pragmos_synthetic_longmemeval_500_v1_manifest.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--session-min", type=int, default=DEFAULT_SESSION_MIN)
    parser.add_argument("--session-max", type=int, default=DEFAULT_SESSION_MAX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile_data)
    official_records = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(official_records, list):
        raise ValueError("Profile data must be a JSON list.")
    profile = profile_from_official(official_records)
    records, family_counts = generate_records(
        profile,
        seed=args.seed,
        session_min=args.session_min,
        session_max=args.session_max,
    )
    validation = validate_records(
        records,
        profile,
        official_records=official_records,
        session_min=args.session_min,
        session_max=args.session_max,
    )
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    manifest = write_artifacts(
        records,
        family_counts,
        validation,
        output_path=output_path,
        manifest_path=manifest_path,
        profile_path=profile_path,
        seed=args.seed,
        session_min=args.session_min,
        session_max=args.session_max,
    )
    print(f"Wrote synthetic dataset: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Records: {validation['record_count']}")
    print(f"Sessions: {validation['session_count']['total']}")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")


if __name__ == "__main__":
    main()
