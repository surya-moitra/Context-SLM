"""Immutable, source-indexed claims for the opt-in typed computation protocol.

Lexers parse source syntax, never question templates. Semantic role choices are
made by the local model and retained for audit; source grounding is not semantic
equivalence verification. The V3 lexer, actor rules and graph are not modified.
"""

from dataclasses import dataclass
import calendar
import datetime as dt
from decimal import DecimalException, localcontext
from functools import wraps
import hashlib
import json
import re

from PRAGMOS_composable_execution import (
    ContractError, Quantity, _combine_quantities, decimal, normalized,
    source_measurements, unit,
)
from PRAGMOS_grounded_computation import evidence_units


SELF = {"i", "my", "me", "mine", "we", "our", "ours", "i'm", "i've", "we're"}
NUMERIC = re.compile(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)")
MONTHS = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold(): i for i, name in enumerate(calendar.month_abbr) if name})
CALENDAR_TEXT = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\s+(?:" + "|".join(MONTHS) +
    r")\s+\d{4}|(?:" + "|".join(MONTHS) + r")\s+\d{1,2},?\s+\d{4}|(?:" +
    "|".join(MONTHS) + r")\s+\d{4})\b", re.IGNORECASE,
)
CLOCK_TEXT = re.compile(r"(?<![\w:])(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*[APap][Mm])?(?![\w:])")


def source_clauses(text):
    """Source syntax only. Keep qualifications available, but not interchangeable."""
    return tuple(part.strip() for part in re.split(r"[;\n]|(?<=[.!?])\s+", text) if part.strip())


def identity_text(value):
    return " ".join(word for word in normalized(value).split() if word not in {"the", "a", "an", "my", "our", "that", "this"})


def literal_labels(text, labels):
    """Bind an operator label to an actual word, not a generated interpretation."""
    words = {word.strip(".,:;?!()[]\"'") for word in normalized(text).split()}
    return tuple(label for label in labels if label in words)


def weekday_literals(text):
    """A source datatype lexer, independent of the question or benchmark."""
    names = {name.casefold() for name in calendar.day_name}
    return tuple(dict.fromkeys(match.group() for match in re.finditer(r"\b[A-Za-z]+\b", text)
                               if match.group().casefold() in names))


@dataclass(frozen=True)
class SourceRelation:
    subject: str
    predicate: str
    object: str
    quote: str


def bind_relation(window, subject, predicate, object_value):
    for value in (subject, predicate, object_value):
        window.literal(value)
    if not all((subject, predicate, object_value)):
        raise ContractError("relation_requires_three_source_arguments")
    clauses = [part for part in source_clauses(window.text) if all(value in part for value in (subject, predicate, object_value))]
    if len(clauses) != 1:
        raise ContractError("relation_arguments_not_in_one_source_assertion")
    clause = clauses[0]
    a, p, b = clause.find(subject), clause.find(predicate), clause.find(object_value)
    if (not a + len(subject) <= p or not p + len(predicate) <= b
            or clause[a + len(subject):p].strip(" ,:") or clause[p + len(predicate):b].strip(" ,:")):
        raise ContractError("source_relation_argument_order_unbound")
    if set(normalized(predicate).split()).intersection({"and", "but", "where", "whereas", "while", "which", "who", "whose"}):
        raise ContractError("relationship_predicate_crosses_another_assertion")
    if set(re.findall(r"\b\w+\b", normalized(clause[a:b]))).intersection({"not", "never", "no"}):
        raise ContractError("negated_relation_cannot_support_positive_projection")
    return SourceRelation(subject, predicate, object_value, clause)


def inferred_relation(window, subject, object_value):
    """Only an unambiguous local ordered relation; context cannot donate an object."""
    if not subject or not object_value or subject == object_value:
        return None
    matches = []
    for clause in (part.strip() for part in re.split(r"[;,\n]", window.text)):
        if clause.count(subject) == clause.count(object_value) == 1:
            a = clause.index(subject) + len(subject)
            b = clause.index(object_value)
            predicate = clause[a:b].strip(" ,:")
            if a < b and predicate:
                matches.append(bind_relation(window, subject, predicate, object_value))
    return matches[0] if len(matches) == 1 else None


def missing_qualification_supported(window, selection):
    quote = selection["missing_span"]
    clauses = [part for part in source_clauses(window.text) if quote and quote in part]
    if len(clauses) != 1:
        return False
    clause = clauses[0]
    words = set(re.findall(r"\b\w+\b", normalized(quote)))
    explicit = bool(words.intersection({"unknown", "unrecorded", "unreported", "unavailable", "unspecified"}))
    explicit |= bool({"not", "recorded"}.issubset(words) or {"not", "known"}.issubset(words))
    if not explicit:
        return False
    # The qualification must concern this fact, not a neighbouring assertion.
    identity = selection["entity"]
    attribute = selection["attribute"]
    parts = source_clauses(window.text)
    previous = parts[parts.index(clause) - 1] if parts.index(clause) else ""
    explicit_reference = normalized(clause).split()[:1] in (["its"], ["their"])
    return bool((identity and identity in clause) or (attribute and attribute in clause)
                or len(parts) == 1 or (explicit_reference and identity and identity in previous))


def uniformity_supported(claim, window):
    quote = json.loads(claim.selection_json).get("uniformity")
    return _uniform_price_assertion(window, quote, claim.entity)


def uniformity_quote(window, entity):
    candidates = [part.strip() for part in re.split(r"[;,\n]|(?<=[.!?])\s+", window.text) if part.strip()]
    candidates = [quote for quote in candidates if _uniform_price_assertion(window, quote, entity)]
    return candidates[0] if len(candidates) == 1 else None


def _uniform_price_assertion(window, quote, entity):
    if not quote:
        return False
    # This recognizes source equality assertions, never question templates.
    # Equality of another property (colour, size, etc.) cannot certify price.
    for clause in (part.strip() for part in re.split(r"[;,\n]|(?<=[.!?])\s+", window.text)):
        if quote not in clause:
            continue
        words = re.findall(r"\b\w+\b", normalized(clause))
        if set(words).intersection({"not", "never", "different", "unequal"}):
            continue
        equality = {"same", "equal", "identical", "uniform"}.intersection(words)
        quantifier = {"all", "each", "every"}.intersection(words)
        money_property = {"price", "prices", "cost", "costs"}.intersection(words)
        entity_words = {count_dimension(word) for word in identity_text(entity).split()}
        local_words = {count_dimension(word) for word in words}
        implicit = words and words[0] in {"all", "each", "every"}
        grammar = {"all", "each", "every", "at", "were", "are", "was", "is", "had", "have", "has", "for", "the", "a", "same", "equal", "identical", "uniform", "price", "prices", "cost", "costs", "of", "one"}
        unknown_nouns = {count_dimension(word) for word in words if word not in grammar} - entity_words
        group_bound = (bool(entity_words & local_words) or implicit) and not unknown_nouns
        if equality and quantifier and money_property and group_bound and any(word in normalized(quote).split() for word in equality):
            return True
    return False


@dataclass(frozen=True)
class QuantityBounds:
    lower: object
    upper: object
    lower_closed: bool = True
    upper_closed: bool = True

    def as_dict(self):
        return {"lower": self.lower.as_dict() if self.lower else None,
                "upper": self.upper.as_dict() if self.upper else None,
                "lower_closed": self.lower_closed, "upper_closed": self.upper_closed}


def source_bound(text, slot, value, requested):
    prefix = normalized(text[:slot.start]).split()[-4:]
    if any(word in prefix for word in ("possibly", "perhaps", "maybe", "about", "roughly", "approximately")):
        raise ContractError("possible_value_is_not_a_certified_bound")
    tail = " ".join(prefix[-2:])
    actual = "lower" if tail in {"at least", "no less"} else "upper" if tail in {"at most", "no more"} else None
    if actual != requested:
        raise ContractError("selected_bound_not_stated_in_source")
    return QuantityBounds(value if actual == "lower" else None, value if actual == "upper" else None)


@dataclass(frozen=True)
class TemporalRelation:
    reference: str
    direction: str
    amount: int | None = None
    unit: str | None = None


def bind_temporal_relation(window, source, selection):
    from PRAGMOS_temporal_grounding import describe_time
    value = describe_time(window.text, source.timestamp)
    relation = value.get("relation")
    if relation:
        return TemporalRelation(relation["reference_text"], relation["direction"], int(relation["amount"]), relation["unit"])
    anchor, direction = selection.get("time_anchor"), selection.get("time_direction")
    if anchor or direction:
        window.literal(anchor)
        if not anchor or direction not in {"before", "after"}:
            raise ContractError("invalid_source_temporal_constraint")
        clauses = [part for part in source_clauses(window.text) if anchor in part and direction in normalized(part).split()]
        if len(clauses) != 1:
            raise ContractError("temporal_constraint_not_bound_to_source_assertion")
        return TemporalRelation(anchor, direction)
    return None


def bounded_decimal_context(function):
    """Keep experimental arithmetic independent of process-global precision."""
    @wraps(function)
    def execute(*args, **kwargs):
        try:
            with localcontext() as context:
                context.prec = 60
                return function(*args, **kwargs)
        except DecimalException as exc:
            raise ContractError("typed_decimal_calculation_not_exact_or_representable") from exc
    return execute


@dataclass(frozen=True)
class CalendarValue:
    start: dt.date
    end: dt.date
    granularity: str
    raw: str

    def contains(self, other):
        return self.start <= other.start and other.end <= self.end


def calendar_value(text):
    text = str(text or "").strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%B %d, %Y",
                "%b %d %Y", "%b %d, %Y"):
        try:
            day = dt.datetime.strptime(text, fmt).date()
            return CalendarValue(day, day, "day", text)
        except ValueError:
            pass
    words = text.casefold().split()
    if len(words) == 2 and words[0] in MONTHS and words[1].isdigit():
        year, month = int(words[1]), MONTHS[words[0]]
        return CalendarValue(dt.date(year, month, 1), dt.date(year, month, calendar.monthrange(year, month)[1]), "month", text)
    if len(text) == 4 and text.isdigit() and 1 <= int(text) <= 9999:
        return CalendarValue(dt.date(int(text), 1, 1), dt.date(int(text), 12, 31), "year", text)
    raise ContractError("unsupported_or_ambiguous_calendar_literal")


def count_dimension(noun):
    value = normalized(noun)
    if not value:
        raise ContractError("count_requires_source_noun")
    # Conservative inflection normalization, not an entity/alias guess.
    if value.endswith("s") and not value.endswith(("ss", "us")):
        value = value[:-1]
    return "count:" + value


@dataclass(frozen=True)
class Window:
    window_id: str
    source_id: str
    start: int
    end: int
    context_start: int
    context_end: int
    text: str
    context: str

    def literal(self, value):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ContractError("empty_typed_source_literal")
        positions = [match.start() for match in re.finditer(re.escape(value), self.context)
                     if (match.start() == 0 or not self.context[match.start() - 1].isalnum())
                     and (match.end() == len(self.context) or not self.context[match.end()].isalnum())]
        if not positions:
            raise ContractError("typed_literal_not_in_source_window")
        # Repeated words are allowed; a role literal is not itself a unique event.
        return value


def source_windows(sources):
    units = evidence_units(sources)
    result = []
    for i, focus in enumerate(units):
        start, end = focus.start, focus.end
        if i and units[i - 1].source_id == focus.source_id:
            start = units[i - 1].start
        if i + 1 < len(units) and units[i + 1].source_id == focus.source_id:
            end = units[i + 1].end
        source = sources[focus.source_id]
        result.append(Window(focus.unit_id, focus.source_id, focus.start, focus.end,
                             start, end, focus.text, source.text[start:end]))
    return result


def span_choices(text, width=6):
    tokens = list(re.finditer(r"\b[\w'-]+\b", text))
    return tuple(sorted({text[tokens[a].start():tokens[b].end()]
                         for a in range(len(tokens)) for b in range(a, min(len(tokens), a + width))}))


def actor_choices(text):
    """Grammatical actor literals, excluding action phrases such as 'I've read'."""
    choices = []
    for span in span_choices(text, width=4):
        words = span.split()
        if normalized(span) in {"there", "the", "a", "an", "for", "every", "each", "all", "on", "at", "in"}:
            continue
        if (len(words) == 1 and (normalized(words[0]) in SELF or words[0][:1].isupper())) or (
            normalized(words[0]) in {"my", "our"} and len(words) > 1
        ) or all(word[:1].isupper() for word in words):
            choices.append(span)
    return tuple(choices)


def measurement_groups(text):
    """Join adjacent components of source measurement syntax, not whole events."""
    slots = numeric_slots(text)
    groups = []
    for index, slot in enumerate(slots):
        joined = False
        if groups and slot.raw_unit:
            previous = slots[groups[-1][-1]]
            gap = normalized(text[previous.end:slot.start]).strip(", ")
            if previous.raw_unit and gap in {"", "and"} and unit(previous.raw_unit).dimensions == unit(slot.raw_unit).dimensions:
                groups[-1].append(index)
                joined = True
        if not joined:
            groups.append([index])
    return groups


@dataclass(frozen=True)
class NumberSlot:
    start: int
    end: int
    text: str
    value: str
    raw_unit: str | None


def numeric_slots(text):
    """Source-only number positions, preserving known units and date exclusions."""
    dates = [(match.start(), match.end()) for parser in (CALENDAR_TEXT, CLOCK_TEXT) for match in parser.finditer(text)]
    result = []
    for measurement in source_measurements(text):
        for match in re.finditer(re.escape(measurement["text"]), text):
            if not any(a <= match.start() < b for a, b in dates):
                result.append(NumberSlot(match.start(), match.end(), match.group(), measurement["value"], measurement["unit"]))
    for match in NUMERIC.finditer(text):
        if any(a <= match.start() < b for a, b in dates) or any(s.start <= match.start() < s.end for s in result):
            continue
        result.append(NumberSlot(match.start(), match.end(), match.group(), str(decimal(match.group())), None))
    # Also preserve spelled-out cardinals without guessing an adjacent noun's type.
    from PRAGMOS_temporal_grounding import CARDINALS
    from PRAGMOS_composable_execution import _measurement_number
    for match in re.finditer(r"\b[A-Za-z]+(?:[- ][A-Za-z]+)*\b", text):
        words = list(re.finditer(r"[A-Za-z]+", match.group()))
        i = 0
        while i < len(words):
            if words[i].group().casefold() not in CARDINALS:
                i += 1
                continue
            stop = i + 1
            while stop < len(words) and words[stop].group().casefold() in set(CARDINALS) | {"hundred", "thousand"}:
                stop += 1
            a, b = match.start() + words[i].start(), match.start() + words[stop - 1].end()
            if not any(s.start <= a < s.end for s in result):
                raw = text[a:b]
                result.append(NumberSlot(a, b, raw, str(_measurement_number(raw)), None))
            i = stop
    return tuple(sorted(set(result), key=lambda slot: (slot.start, slot.end)))


def _bounded_literal(text, slot):
    prefix = normalized(text[:slot.start]).split()[-4:]
    suffix = text[slot.end:].lstrip()
    if set(prefix).intersection({"about", "approximately", "roughly", "around", "least", "most", "over", "under", "possibly", "perhaps", "maybe"}):
        raise ContractError("typed_quantity_requires_interval_contract")
    if re.search(r"\d\s*(?:-|to)\s*$", text[:slot.start]) or re.match(r"(?:-|to\b)\s*\d", suffix):
        raise ContractError("typed_quantity_range_not_exact")


@bounded_decimal_context
def bind_quantity(window, indexes, kind, attribute, *, allow_bound=False):
    if not isinstance(indexes, list) or len(indexes) > 4 or len(set(indexes)) != len(indexes):
        raise ContractError("invalid_typed_number_selection")
    if not indexes:
        if kind not in {"none", "missing"}:
            raise ContractError("typed_quantity_missing_number")
        return None
    slots = numeric_slots(window.text)
    values = []
    for index in indexes:
        if type(index) is not int or not 0 <= index < len(slots):
            raise ContractError("typed_number_index_out_of_range")
        slot = slots[index]
        if not allow_bound:
            _bounded_literal(window.text, slot)
        if slot.raw_unit:
            if kind not in {"measure", "rate"}:
                raise ContractError("stated_unit_cannot_be_reinterpreted")
            value = Quantity.from_raw(slot.value, slot.raw_unit)
        elif kind == "count":
            if not attribute:
                raise ContractError("typed_count_requires_source_attribute")
            value = Quantity(decimal(slot.value), ((count_dimension(attribute), 1),), count_dimension(attribute))
        elif kind == "scalar":
            if not attribute:
                raise ContractError("bare_scalar_requires_source_property")
            label = "scalar:" + normalized(attribute)
            value = Quantity(decimal(slot.value), ((label, 1),), label)
        else:
            raise ContractError("bare_number_requires_typed_binding")
        if kind == "rate":
            if not attribute or not set(re.findall(r"\b\w+\b", normalized(window.text))).intersection({"per", "each", "every"}):
                raise ContractError("rate_requires_source_denominator")
            from PRAGMOS_composable_execution import dimensions
            value = Quantity(value.amount, dimensions(value.dimensions, ((count_dimension(attribute), 1),), -1), value.display_unit)
        values.append(value)
    result = values[0]
    for value in values[1:]:
        result = _combine_quantities(result, value, "add")
    return result


@dataclass(frozen=True)
class TypedClaim:
    claim_id: str
    window_id: str
    source_id: str
    selection_json: str
    source_speaker: str
    actor: str
    actor_binding: str
    entity: str | None
    key: str | None
    attribute: str | None
    operand: str
    group_side: str
    kind: str
    status: str
    change: str
    quantity: Quantity | None
    occurred_at: CalendarValue | None
    link: str | None
    relation: SourceRelation | None = None
    bounds: QuantityBounds | None = None
    event: str | None = None
    temporal_relation: TemporalRelation | None = None
    clock_minutes: int | None = None


FIELDS = {"actor", "binding", "relation_anchor", "entity", "key", "attribute", "numbers", "quantity_kind",
          "operand", "group_side", "kind", "status", "change", "time", "link", "missing_span", "uniformity",
          "subject", "predicate", "object", "event", "bound", "time_anchor", "time_direction", "clock"}


def bind_typed_claim(window, source, selection, known):
    if not isinstance(selection, dict) or set(selection) != FIELDS:
        raise ContractError("invalid_typed_claim_fields")
    if source.text[window.start:window.end] != window.text or source.text[window.context_start:window.context_end] != window.context:
        raise ContractError("typed_source_window_was_modified")
    for name in ("actor", "relation_anchor", "entity", "key", "attribute", "time", "missing_span", "uniformity",
                 "subject", "predicate", "object", "event", "time_anchor", "clock"):
        window.literal(selection[name])
    for name, allowed in (
        ("binding", {"self", "requested_relation", "linked_event", "named", "event_property", "other", "unknown"}),
        ("operand", {"left", "right", "member", "context"}),
        ("group_side", {"left", "right", "none"}),
        ("kind", {"assertion", "reference", "context", "unknown"}),
        ("status", {"completed", "ongoing", "pending", "planned", "negated", "unknown"}),
        ("change", {"present", "absent", "none", "unknown"}),
        ("quantity_kind", {"measure", "count", "scalar", "rate", "none", "missing"}),
    ):
        if selection[name] not in allowed:
            raise ContractError("invalid_typed_" + name)
    if "unknown" in (selection["binding"], selection["kind"], selection["status"], selection["change"]):
        raise ContractError("unresolved_typed_semantics")
    actor_span = selection["actor"]
    binding = selection["binding"]
    parent = known.get(selection["link"]) if selection["link"] else None
    if selection["link"] and parent is None:
        raise ContractError("unresolved_typed_reference")
    if selection["kind"] == "reference" and parent is None:
        raise ContractError("repeated_fact_requires_a_grounded_antecedent_not_a_repeated_entity")
    if binding == "self" and normalized(actor_span) not in SELF:
        raise ContractError("typed_actor_is_not_source_speaker")
    if binding == "requested_relation":
        anchor = normalized(selection["relation_anchor"])
        if not actor_span or not set(anchor.split()).intersection({"my", "our", "mine", "ours"}):
            raise ContractError("related_actor_requires_possessive_source_link")
    if binding == "linked_event" and parent is None:
        raise ContractError("linked_event_requires_grounded_parent")
    if binding == "linked_event" and actor_span:
        anchors = {normalized(parent.actor), normalized(parent.entity), normalized(parent.key)}
        if normalized(actor_span) not in anchors | {"it", "this", "that", "they"} and not (
            normalized(actor_span) in SELF and normalized(parent.actor) == normalized(source.speaker)
        ):
            raise ContractError("linked_event_cannot_replace_an_explicit_actor")
    if binding == "named" and (not actor_span or normalized(actor_span) in SELF):
        raise ContractError("named_actor_requires_explicit_source_person")
    if binding == "event_property" and (not selection["event"] or actor_span):
        raise ContractError("event_property_requires_an_event_not_a_guessed_actor")
    if not actor_span and parent is None and binding != "event_property":
        raise ContractError("typed_actor_not_bound")
    actor = source.speaker if binding in {"self", "event_property"} else parent.actor if binding == "linked_event" else actor_span
    if parent is not None and binding not in {"linked_event", "self"} and normalized(actor) != normalized(parent.actor):
        raise ContractError("typed_reference_crosses_actors")
    if parent is not None:
        reference = normalized(selection["entity"])
        names = {normalized(parent.entity), normalized(parent.key)} - {""}
        head = identity_text(selection["entity"]).split()[-1:]
        head_matches = [c for c in known.values() if head and head == identity_text(c.entity).split()[-1:]]
        unique_head_reference = (len(head_matches) == 1 and head_matches[0].claim_id == parent.claim_id
                                 and bool({"that", "this", "these", "those"}.intersection(normalized(window.text).split())))
        noun = selection["attribute"]
        denominator = count_dimension(noun) if noun else None
        count_parents = [c for c in known.values() if c.quantity and c.source_speaker == source.speaker
                         and c.quantity.dimensions == ((denominator, 1),)] if denominator else []
        quantified_member_reference = (selection["quantity_kind"] == "rate" and
            bool(literal_labels(window.text, ("every", "each", "per"))) and len(count_parents) == 1
            and count_parents[0].claim_id == parent.claim_id)
        if reference not in names and not reference.split()[:1] in (
            ["it"], ["this"], ["that"], ["these"], ["those"], ["same"],
        ) and not any(name in normalized(window.context) for name in names) and not unique_head_reference and not quantified_member_reference:
            raise ContractError("typed_reference_has_no_source_link")
    if selection["quantity_kind"] == "missing" and not selection["missing_span"]:
        raise ContractError("missing_value_requires_source_qualification")
    if selection["quantity_kind"] == "missing" and selection["missing_span"] not in window.text:
        raise ContractError("missing_value_qualification_not_bound_to_focus")
    if selection["quantity_kind"] == "missing" and not missing_qualification_supported(window, selection):
        raise ContractError("missing_value_requires_explicit_fact_bound_uncertainty")
    if selection["time"] and selection["time"] not in window.text:
        dates = {match.group().casefold() for match in CALENDAR_TEXT.finditer(window.context)}
        if len(dates) != 1:
            raise ContractError("ambiguous_context_calendar_binding")
    bound = selection["bound"]
    if bound not in {None, "lower", "upper"}:
        raise ContractError("invalid_quantity_bound")
    quantity = bind_quantity(window, selection["numbers"], selection["quantity_kind"], selection["attribute"], allow_bound=bool(bound))
    bounds = None
    if bound:
        if quantity is None or len(selection["numbers"]) != 1:
            raise ContractError("bound_requires_one_grounded_measurement")
        bounds = source_bound(window.text, numeric_slots(window.text)[selection["numbers"][0]], quantity, bound)
    if selection["quantity_kind"] == "missing" and quantity is not None:
        raise ContractError("missing_value_cannot_supply_number")
    occurred_at = calendar_value(selection["time"]) if selection["time"] else None
    reference_words = normalized(selection["entity"]).split()[:1]
    entity = parent.entity if parent is not None and reference_words in (["it"], ["this"], ["that"], ["these"], ["those"], ["same"]) else selection["entity"]
    key = selection["key"] or (parent.key if parent is not None else None)
    if not entity and not key and quantity is None and occurred_at is None:
        raise ContractError("typed_claim_has_no_bound_argument")
    payload = json.dumps(selection, sort_keys=True)
    identifier = hashlib.sha256((source.source_id + ":" + str(window.start) + ":" + payload).encode()).hexdigest()[:20]
    relation = (bind_relation(window, selection["subject"], selection["predicate"], selection["object"])
                if any(selection[name] for name in ("subject", "predicate", "object")) else inferred_relation(window, entity, key))
    temporal_relation = bind_temporal_relation(window, source, selection)
    clock = None
    if selection["clock"]:
        match = CLOCK_TEXT.fullmatch(selection["clock"])
        if not match:
            raise ContractError("invalid_source_clock_value")
        raw = selection["clock"].strip().casefold()
        meridiem = raw[-2:] if raw[-2:] in {"am", "pm"} else None
        hour, minute = (int(part) for part in (raw[:-2].strip() if meridiem else raw).split(":"))
        if meridiem and not 1 <= hour <= 12:
            raise ContractError("invalid_meridiem_clock_value")
        clock = (hour % 12 + (12 if meridiem == "pm" else 0)) * 60 + minute if meridiem else hour * 60 + minute
    return TypedClaim(identifier, window.window_id, source.source_id, payload, source.speaker, actor, binding, entity, key,
                      selection["attribute"], selection["operand"], selection["group_side"], selection["kind"],
                      selection["status"], selection["change"], quantity, occurred_at, parent.claim_id if parent else None,
                      relation, bounds, selection["event"], temporal_relation, clock)


def validate_claims(claims, windows, sources):
    known = {}
    for claim in claims:
        reconstructed = bind_typed_claim(windows[claim.window_id], sources[claim.source_id],
                                          json.loads(claim.selection_json), known)
        if reconstructed != claim:
            raise ContractError("typed_claim_not_equal_to_source_binding")
        known[claim.claim_id] = claim
    return known
