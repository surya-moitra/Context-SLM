"""Source-backed, bounded relational execution for local memory reasoning.

This module knows nothing about benchmark labels or gold answers. Existing
inference remains unchanged unless its caller explicitly enables the candidate.
"""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import re


VERSION = 1
MAX_NODES = 16
MAX_DEPTH = 8
MAX_FACTS = 256
MAX_ROWS = 512
MAX_SOURCES = 32
ROLES = frozenset({"subject", "object", "instrument", "theme", "location", "label",
                   "event", "person", "weekday", "time"})
STATUSES = frozenset({"completed", "ongoing", "planned", "pending", "negated", "unknown"})
OPS = frozenset({"source", "filter", "project", "distinct", "reduce", "join",
                 "union", "intersection", "arithmetic", "calculate", "rank", "state_fold"})


class ContractError(ValueError):
    pass


def normalized(value):
    return " ".join(str(value or "").casefold().split())


def decimal(value):
    if isinstance(value, bool) or value is None:
        raise ContractError("missing_or_invalid_number")
    try:
        result = Decimal(str(value).replace(",", ""))
    except InvalidOperation as exc:
        raise ContractError("invalid_number") from exc
    if not result.is_finite() or abs(result) > Decimal("1e30"):
        raise ContractError("nonfinite_or_excessive_number")
    return result


@dataclass(frozen=True)
class Unit:
    dimensions: tuple
    factor: Decimal
    label: str


_UNIT_GROUPS = (
    ("second", "duration", "1", ("second", "seconds", "sec", "s")),
    ("minute", "duration", "60", ("minute", "minutes", "min", "mins")),
    ("hour", "duration", "3600", ("hour", "hours", "hr", "hrs")),
    ("day", "duration", "86400", ("day", "days")),
    ("week", "duration", "604800", ("week", "weeks")),
    ("month", "calendar_month", "1", ("month", "months")),
    ("year", "calendar_year", "1", ("year", "years")),
    ("g", "mass", "1", ("g", "gram", "grams")),
    ("kg", "mass", "1000", ("kg", "kilogram", "kilograms")),
    ("m", "length", "1", ("m", "meter", "meters", "metre", "metres")),
    ("km", "length", "1000", ("km", "kilometer", "kilometers", "kilometre", "kilometres")),
    ("cm", "length", ".01", ("cm", "centimeter", "centimeters", "centimetre", "centimetres")),
    ("usd", "currency:usd", "1", ("usd", "$", "dollar", "dollars", "money", "currency")),
    ("eur", "currency:eur", "1", ("eur", "euro", "euros")),
    ("gbp", "currency:gbp", "1", ("gbp", "pound sterling", "pounds sterling")),
    ("item", "count", "1", ("item", "items")),
    ("page", "pages", "1", ("page", "pages")),
    ("percent", None, ".01", ("percent", "percentage", "%")),
    ("1", None, "1", ("1", "number", "dimensionless")),
)
_UNITS = {alias: Unit(((dimension, 1),) if dimension else (), Decimal(factor), label)
          for label, dimension, factor, aliases in _UNIT_GROUPS for alias in aliases}


def dimensions(left, right, sign=1):
    values = defaultdict(int, left)
    for key, exponent in right:
        values[key] += exponent * sign
    return tuple(sorted((key, value) for key, value in values.items() if value))


def unit(value):
    name = normalized(value)
    if name in _UNITS:
        return _UNITS[name]
    # This parses unit syntax, never question text or benchmark categories.
    parts = name.replace(" per ", "/").split("/")
    if len(parts) == 2:
        left, right = unit(parts[0]), unit(parts[1])
        return Unit(dimensions(left.dimensions, right.dimensions, -1),
                    left.factor / right.factor, f"{left.label}/{right.label}")
    raise ContractError("unknown_measurement_unit:" + name)


@dataclass(frozen=True)
class Quantity:
    amount: Decimal                 # Canonical units; raw input remains in Fact.
    dimensions: tuple
    display_unit: str

    @classmethod
    def from_raw(cls, value, raw_unit):
        definition = unit(raw_unit)
        return cls(decimal(value) * definition.factor, definition.dimensions, definition.label)

    def convert(self, target):
        definition = unit(target)
        if definition.dimensions != self.dimensions:
            raise ContractError("incompatible_units")
        return self.amount / definition.factor

    def as_dict(self):
        return {"canonical_value": str(self.amount), "dimensions": list(self.dimensions),
                "display_unit": self.display_unit}


@dataclass(frozen=True)
class Source:
    source_id: str
    session_id: str
    turn_id: str
    role: str
    speaker: str
    timestamp: str | None
    text: str

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class AliasBinding:
    role: str
    alias: str
    canonical: str
    source_id: str
    quote: str
    predicate: str


def grounded_aliases(bindings, sources):
    result, provenance = {}, []
    if isinstance(bindings, dict):
        raise ContractError("alias_source_bindings_required")
    for binding in bindings or ():
        if not isinstance(binding, AliasBinding) or binding.role not in ROLES:
            raise ContractError("invalid_alias_binding")
        source = sources.get(binding.source_id)
        if source is None:
            raise ContractError("unresolved_alias_source")
        _source_span(source.text, binding.quote)
        for value in (binding.alias, binding.canonical, binding.predicate):
            _supported_span(value, binding.quote)
        if normalized(binding.predicate) not in {"call", "called", "known as", "also named", "renamed", "goes by"}:
            raise ContractError("alias_identity_relation_required")
        if set(re.findall(r"[a-z]+", normalized(binding.quote))).intersection({"not", "never", "isnt", "wasnt"}):
            raise ContractError("negated_alias_identity")
        key = (binding.role, normalized(binding.alias))
        if key in result and normalized(result[key]) != normalized(binding.canonical):
            raise ContractError("ambiguous_alias_identity")
        result[key] = binding.canonical
        provenance.append({**binding.__dict__, "source": source.as_dict()})
    return result, provenance


@dataclass(frozen=True)
class Fact:
    fact_id: str
    source_id: str
    quote: str
    start: int
    end: int
    actor: str
    actor_span: str
    predicate: str
    status: str
    kind: str
    operands: tuple
    roles: tuple
    event_id: str
    quantity: Quantity | None
    raw_value: str | None
    raw_unit: str | None
    raw_measurement: str | None

    def as_dict(self):
        return {**self.__dict__, "roles": dict(self.roles),
                "quantity": self.quantity.as_dict() if self.quantity else None,
                "normalization": ({"source_unit": self.raw_unit,
                                   "factor": str(unit(self.raw_unit).factor),
                                   "canonical_value": str(self.quantity.amount)}
                                  if self.quantity else None)}


@dataclass(frozen=True)
class SelectedFact(Fact):
    selection_json: str
    change: str


def _source_span(text, quote, start=None):
    if not isinstance(quote, str) or not quote.strip():
        raise ContractError("missing_source_quote")
    if start is None:
        if text.count(quote) != 1:
            raise ContractError("ambiguous_or_invented_source_span")
        start = text.index(quote)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0 or text[start:start + len(quote)] != quote:
        raise ContractError("invalid_source_span")
    return start, start + len(quote)


def _supported_span(value, quote):
    value = str(value or "").strip()
    # Exact spans retain qualifiers and prevent ungrounded semantic labels.
    if not value or value not in quote:
        raise ContractError("invented_argument_span")
    positions = [match.start() for match in re.finditer(re.escape(value), quote)]
    if not any(
        (not value[0].isalnum() or start == 0 or not quote[start - 1].isalnum())
        and (not value[-1].isalnum() or start + len(value) == len(quote)
             or not quote[start + len(value)].isalnum()) for start in positions
    ):
        raise ContractError("argument_is_only_a_word_fragment")
    return value


def _measurement_number(text):
    mentions = re.findall(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?![\d])", text)
    if len(mentions) == 1:
        return decimal(mentions[0])
    if mentions:
        raise ContractError("ambiguous_raw_measurement")
    from PRAGMOS_temporal_grounding import CARDINALS
    tokens = re.findall(r"[a-z]+", normalized(text))
    total, current, consumed = 0, 0, False
    for index, token in enumerate(tokens):
        if token in CARDINALS:
            current += CARDINALS[token]
        elif token == "hundred":
            current = max(1, current) * 100
        elif token == "thousand":
            total, current = total + max(1, current) * 1000, 0
        elif token == "and" and consumed and index + 1 < len(tokens) and tokens[index + 1] in CARDINALS:
            continue
        else:
            break
        consumed = True
    if not consumed:
        raise ContractError("ungrounded_raw_measurement")
    return Decimal(total + current)


def source_measurements(text):
    """Lex explicit adjacent number/unit spans, independently of any question."""
    from PRAGMOS_temporal_grounding import CARDINALS
    tokens = list(re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[A-Za-z]+|[$%]", text))
    result = []
    for index, token in enumerate(tokens):
        word = token.group().casefold()
        if not (word in CARDINALS or word[:1].isdigit() or (word[:1] in {"-", "+"} and word[1:2].isdigit())):
            continue
        if index and token.start() == tokens[index - 1].end() and tokens[index - 1].group().isalpha():
            continue
        end = index + 1
        if word in CARDINALS:
            if index and tokens[index - 1].group().casefold() in CARDINALS | {"hundred": 100, "thousand": 1000}:
                continue
            while end < len(tokens):
                following = tokens[end].group().casefold()
                if following in CARDINALS or following in {"hundred", "thousand"}:
                    end += 1
                elif following == "and" and end + 1 < len(tokens) and tokens[end + 1].group().casefold() in CARDINALS:
                    end += 1
                else:
                    break
        raw_number = text[token.start():tokens[end - 1].end()]
        definition, unit_name, finish = None, None, end
        for width in (3, 2, 1):
            if end + width <= len(tokens):
                candidate = normalized(text[tokens[end].start():tokens[end + width - 1].end()])
                if candidate in _UNITS and candidate not in {"1", "number", "dimensionless"}:
                    definition, unit_name, finish = _UNITS[candidate], candidate, end + width
                    break
        begin = token.start()
        if definition is None and index:
            preceding = normalized(tokens[index - 1].group())
            candidate = _UNITS.get(preceding)
            if candidate and any(key.startswith("currency:") for key, _ in candidate.dimensions):
                definition, unit_name, begin = candidate, preceding, tokens[index - 1].start()
        if definition is None:
            continue
        # Rate syntax keeps its denominator rather than erasing the dimension.
        if finish < len(tokens) and tokens[finish].group().casefold() == "per" and finish + 1 < len(tokens):
            denominator = normalized(tokens[finish + 1].group())
            if denominator in _UNITS:
                unit_name, finish = unit_name + "/" + denominator, finish + 2
        elif finish < len(tokens) and "/" in text[tokens[finish - 1].end():tokens[finish].start()]:
            denominator = normalized(tokens[finish].group())
            if denominator in _UNITS:
                unit_name, finish = unit_name + "/" + denominator, finish + 1
        try:
            value = _measurement_number(raw_number)
        except ContractError:
            continue
        result.append({"text": text[begin:tokens[finish - 1].end()], "value": str(value), "unit": unit_name})
    # The same span can be reached via a cardinal and its trailing number.
    return list({(row["text"], row["value"], row["unit"]): row for row in result}.values())


def parse_measurement_span(text):
    candidates = source_measurements(text)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text.strip()):
        return {"text": text, "value": str(_measurement_number(text)), "unit": "1"}
    raise ContractError("measurement_span_requires_one_explicit_quantity")


def ground_fact(payload, sources, operand_ids):
    source = sources.get(payload.get("source_id"))
    if source is None or not source.session_id or not source.turn_id:
        raise ContractError("unresolved_source")
    quote = payload.get("quote")
    start, end = _source_span(source.text, quote, payload.get("start"))
    actor_span = _supported_span(payload.get("actor_span"), quote)
    predicate = _supported_span(payload.get("predicate"), quote)
    if quote.count(predicate) != 1:
        raise ContractError("ambiguous_predicate_occurrence")
    prefix = quote[:quote.index(predicate)].strip()
    if normalized(prefix) in {"i", "we"}:
        # An explicit active subject is source syntax, not an inferred actor.
        actor_span = prefix
    actor = (source.speaker or source.role) if normalized(actor_span) in {"i", "we"} else actor_span
    status = payload.get("status")
    if status not in STATUSES:
        raise ContractError("invalid_event_status")
    # Reuse the existing source-qualification guard; never mutate its sources.
    from PRAGMOS_evidence_compiler import _authoritative_numeric_claim, derive_event_status_hint
    source_turn = {"session_id": source.session_id, "text": source.text}
    authority = _authoritative_numeric_claim({
        "source_turn_id": source.turn_id, "source_session_id": source.session_id,
        "clause": quote,
    }, {source.turn_id: source_turn})
    source_status = (authority or {}).get("status") or derive_event_status_hint(quote)
    source_status = "negated" if source_status == "negated_or_cancelled" else source_status
    if source_status in {"negated", "planned", "pending"}:
        status = source_status
    kind = payload.get("kind")
    if kind not in {"event", "state", "delta", "reference"}:
        raise ContractError("invalid_assertion_kind")
    members = payload.get("operands")
    if not isinstance(members, list) or not members or not set(members).issubset(operand_ids):
        raise ContractError("unknown_operand_binding")
    roles = payload.get("roles")
    if not isinstance(roles, dict) or set(roles).difference(ROLES):
        raise ContractError("invalid_argument_role")
    roles = tuple(sorted((key, _supported_span(value, quote)) for key, value in roles.items()))
    measure = payload.get("measurement")
    if measure is None:
        explicit = source_measurements(quote)
        if len(explicit) == 1:
            measure = explicit[0]
    elif isinstance(measure, dict) and set(measure) == {"text"}:
        measure = parse_measurement_span(_supported_span(measure["text"], quote))
    quantity, raw_value, raw_unit, measurement = None, None, None, None
    if measure is not None:
        if not isinstance(measure, dict):
            raise ContractError("invalid_measurement")
        measurement = _supported_span(measure.get("text"), quote)
        raw_value, raw_unit = str(measure.get("value")), str(measure.get("unit"))
        number = decimal(raw_value)
        # A lexical numeric parser, not an intent/phrase classifier. One slot
        # holds one explicit measure; compound durations use several slots.
        if quote.count(measurement) != 1:
            raise ContractError("ambiguous_measurement_span")
        if _measurement_number(measurement) != number:
            raise ContractError("ungrounded_raw_measurement")
        raw_value = format(number.normalize(), "f")
        definition = unit(raw_unit)
        stated = normalized(measurement)
        tokens = set(re.findall(r"[a-z]+|[$%]", stated))
        def supports(definition):
            aliases = [alias for alias, candidate in _UNITS.items() if candidate == definition]
            return definition.label == "1" or any(
                alias in tokens or (" " in alias and alias in stated) for alias in aliases
            )
        components = normalized(raw_unit).replace(" per ", "/").split("/")
        supported = supports(definition) if len(components) == 1 else (
            len(components) == 2 and ("per" in tokens or "/" in stated)
            and all(supports(unit(component)) for component in components)
        )
        if not supported:
            raise ContractError("measurement_unit_not_stated")
        if ("per" in tokens or "each" in tokens) and len(components) == 1:
            raise ContractError("rate_denominator_cannot_be_erased")
        if definition.label == "1" and any(
            alias in tokens for alias, candidate in _UNITS.items()
            if candidate.dimensions or candidate.label == "percent"
        ):
            raise ContractError("stated_measurement_unit_cannot_be_erased")
        if definition.label == "1":
            at = quote.index(measurement)
            adjacent = (quote[max(0, at - 1):at], quote[at + len(measurement):])
            following = re.match(r"\s*([a-zA-Z]+|[$%])", adjacent[1])
            adjacent_tokens = {adjacent[0], normalized(following.group(1)) if following else ""}
            if any(alias in adjacent_tokens for alias, candidate in _UNITS.items()
                   if candidate.dimensions or candidate.label == "percent"):
                raise ContractError("stated_measurement_unit_cannot_be_erased")
        prefix = normalized(quote[:quote.index(measurement)]).split()[-4:]
        if set(prefix).intersection({"about", "approximately", "roughly", "around", "least", "most", "over", "under"}):
            raise ContractError("non_exact_measurement_requires_interval_contract")
        suffix = quote[quote.index(measurement) + len(measurement):].lstrip()
        if len(components) == 1 and normalized(suffix).split()[:1] in (["per"], ["each"]):
            raise ContractError("rate_denominator_cannot_be_erased")
        if re.match(r"[.,]\d", suffix):
            raise ContractError("measurement_splits_a_numeric_literal")
        if suffix.startswith(("-", "/")):
            raise ContractError("measurement_range_or_rate_requires_explicit_contract")
        quantity = Quantity.from_raw(number, raw_unit)
    predicate_position = start + quote.index(predicate)
    identity = (source.session_id, source.turn_id, predicate_position, actor)
    event_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    # Different numeric/argument slots in one assertion remain independent.
    measure_position = start + quote.index(measurement) if measurement else None
    slot = (identity, roles, measure_position, measurement, status, kind)
    fact_id = hashlib.sha256(json.dumps(slot).encode()).hexdigest()[:20]
    return Fact(fact_id, source.source_id, quote, start, end, actor, actor_span, predicate, status,
                kind, tuple(sorted(set(members))), roles, event_id, quantity,
                raw_value, raw_unit, measurement)


def _object(value, keys, required=()):
    if not isinstance(value, dict) or set(value).difference(keys) or set(required).difference(value):
        raise ContractError("invalid_contract_fields")


def validate_plan(payload):
    _object(payload, {"version", "applicable", "operands", "steps", "output"},
            {"version", "applicable", "operands", "steps", "output"})
    if type(payload["version"]) is not int or payload["version"] != VERSION or not isinstance(payload["applicable"], bool):
        raise ContractError("invalid_plan_version")
    if not payload["applicable"]:
        raise ContractError("not_applicable")
    operands, steps = payload["operands"], payload["steps"]
    if not isinstance(operands, list) or not 1 <= len(operands) <= 8:
        raise ContractError("operand_budget_exceeded")
    ids = set()
    for operand in operands:
        _object(operand, {"id", "description", "actor", "statuses", "cardinality", "kinds"},
                {"id", "description", "actor", "statuses", "cardinality", "kinds"})
        if not isinstance(operand["id"], str) or not operand["id"] or operand["id"] in ids:
            raise ContractError("duplicate_operand")
        ids.add(operand["id"])
        if not isinstance(operand["description"], str) or not operand["description"].strip():
            raise ContractError("missing_operand_description")
        if not isinstance(operand["actor"], str) or not operand["actor"].strip():
            raise ContractError("missing_actor")
        if operand["cardinality"] not in {"scalar", "collection"}:
            raise ContractError("invalid_operand_cardinality")
        for name, allowed in (("statuses", STATUSES), ("kinds", {"event", "state", "delta", "reference"})):
            if not isinstance(operand[name], list) or not operand[name] or not set(operand[name]).issubset(allowed):
                raise ContractError("invalid_operand_" + name)
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_NODES:
        raise ContractError("plan_budget_exceeded")
    depths, dependencies = {}, {}
    parameters = {
        "source": {"operand"}, "filter": {"where"}, "project": {"columns"},
        "distinct": {"keys"}, "reduce": {"keys", "aggregate", "column"},
        "join": {"on", "cardinality"}, "union": {"keys"}, "intersection": {"keys"},
        "arithmetic": {"operator", "left", "right"}, "rank": {"column", "direction"},
        "calculate": {"operator", "left", "right", "column"},
        "state_fold": {"keys"},
    }
    for step in steps:
        _object(step, {"id", "op", "inputs", "params"}, {"id", "op", "inputs", "params"})
        key, op, inputs, params = step["id"], step["op"], step["inputs"], step["params"]
        if not isinstance(key, str) or not key or key in depths or op not in OPS:
            raise ContractError("invalid_or_duplicate_step")
        if not isinstance(inputs, list) or not all(isinstance(value, str) and value in depths for value in inputs):
            raise ContractError("cycle_or_forward_reference")
        expected = 0 if op == "source" else 2 if op in {"join", "union", "intersection", "arithmetic"} else 1
        if len(inputs) != expected:
            raise ContractError("invalid_step_arity")
        _object(params, parameters[op], parameters[op])
        if op == "source" and params["operand"] not in ids:
            raise ContractError("unknown_source_operand")
        depth = 1 + max((depths[value] for value in inputs), default=0)
        if depth > MAX_DEPTH:
            raise ContractError("plan_depth_exceeded")
        depths[key], dependencies[key] = depth, inputs
    output = payload["output"]
    _object(output, {"step", "columns", "shape", "unit", "decimals"},
            {"step", "columns", "shape", "unit", "decimals"})
    if output["step"] not in depths or output["shape"] not in {"scalar", "value_set", "tuple_set"}:
        raise ContractError("invalid_output_contract")
    if not isinstance(output["columns"], list) or not output["columns"] or not all(
        isinstance(value, str) and value for value in output["columns"]
    ):
        raise ContractError("missing_output_columns")
    if output["shape"] != "tuple_set" and len(output["columns"]) != 1:
        raise ContractError("invalid_output_projection")
    digits = output["decimals"]
    if digits is not None and (isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 8):
        raise ContractError("invalid_output_precision")
    reachable = set()
    def visit(key):
        if key not in reachable:
            reachable.add(key)
            for parent in dependencies[key]:
                visit(parent)
    visit(output["step"])
    if reachable != set(depths):
        raise ContractError("unused_plan_step")
    used_operands = {s["params"]["operand"] for s in steps if s["op"] == "source"}
    if used_operands != ids:
        raise ContractError("unused_mandatory_operand")
    # Return a detached JSON value: callers cannot mutate a validated input in place.
    return json.loads(json.dumps(payload))


@dataclass(frozen=True)
class CellRow:
    cells: tuple
    provenance: frozenset

    def get(self, key):
        values = dict(self.cells)
        if key not in values or values[key] is None or values[key] == "":
            raise ContractError("missing_role:" + key)
        return values[key]


def _cell_key(value):
    if isinstance(value, Quantity):
        return (value.amount, value.dimensions)
    return normalized(value)


def _row_key(row, keys):
    if not isinstance(keys, list) or not keys:
        raise ContractError("missing_identity_projection")
    return tuple(_cell_key(row.get(key)) for key in keys)


def _provenance(rows):
    return frozenset().union(*(row.provenance for row in rows))


def _deduplicate(rows, keys):
    result = {}
    for row in rows:
        identity = _row_key(row, keys)
        if identity in result:
            first = result[identity]
            result[identity] = CellRow(first.cells, first.provenance | row.provenance)
        else:
            result[identity] = row
    return list(result.values())


def _predicate(row, condition, depth=0):
    if depth > 5 or not isinstance(condition, dict):
        raise ContractError("predicate_budget_exceeded")
    if set(condition) in ({"all"}, {"any"}):
        operator = next(iter(condition))
        children = condition[operator]
        if not isinstance(children, list) or not 1 <= len(children) <= 8:
            raise ContractError("invalid_boolean_predicate")
        values = [_predicate(row, child, depth + 1) for child in children]
        return all(values) if operator == "all" else any(values)
    _object(condition, {"field", "operator", "values"}, {"field", "operator", "values"})
    values = condition["values"]
    if not isinstance(values, list) or not values or len(values) > 16:
        raise ContractError("invalid_predicate_values")
    value = row.get(condition["field"])
    operator = condition["operator"]
    if operator in {"in", "not_in"}:
        matched = _cell_key(value) in {_cell_key(v) for v in values}
        return matched if operator == "in" else not matched
    raise ContractError("unsupported_predicate_operator")


def _combine_quantities(left, right, operator):
    if not isinstance(left, Quantity) or not isinstance(right, Quantity):
        raise ContractError("arithmetic_requires_quantities")
    if operator in {"add", "subtract", "absolute_difference"}:
        if left.dimensions != right.dimensions:
            raise ContractError("incompatible_units")
        value = left.amount + right.amount if operator == "add" else left.amount - right.amount
        if operator == "absolute_difference":
            value = abs(value)
        return Quantity(value, left.dimensions, left.display_unit)
    if operator not in {"multiply", "divide"}:
        raise ContractError("unsupported_arithmetic_operator")
    if operator == "divide" and right.amount == 0:
        raise ContractError("zero_denominator")
    power = -1 if operator == "divide" else 1
    dims = dimensions(left.dimensions, right.dimensions, power)
    value = left.amount / right.amount if power == -1 else left.amount * right.amount
    if not dims:
        label = "1"
    elif not right.dimensions:
        label = left.display_unit
    elif power == -1:
        label = f"{left.display_unit}/{right.display_unit}"
    else:
        label = "canonical"
    return Quantity(value, dims, label)


def _execute_step(step, inputs, operands, facts, sources, aliases):
    op, params = step["op"], step["params"]
    rows = inputs[0] if inputs else []
    if op == "source":
        operand = operands[params["operand"]]
        selected = [fact for fact in facts if operand["id"] in fact.operands
                    and normalized(fact.actor) == normalized(operand["actor"])
                    and fact.status in operand["statuses"] and fact.kind in operand["kinds"]]
        if not selected:
            raise ContractError("missing_operand:" + operand["id"])
        if operand["cardinality"] == "scalar" and len(selected) != 1:
            raise ContractError("ambiguous_scalar_operand:" + operand["id"])
        result = []
        for fact in selected:
            values = {"value": fact.quantity, "actor": fact.actor, "status": fact.status,
                      "kind": fact.kind, "predicate": fact.predicate, "event_id": fact.event_id,
                      "fact_id": fact.fact_id, "reported_at": sources[fact.source_id].timestamp}
            if isinstance(fact, SelectedFact):
                source = sources[fact.source_id]
                values["change"] = fact.change
                values["claim_order"] = (str(source.timestamp or ""), str(source.turn_id).zfill(16), fact.start)
            for role, text in fact.roles:
                # Explicit role-scoped aliases only. No mutable context resolver.
                values[role] = aliases.get((role, normalized(text)), text)
            result.append(CellRow(tuple(values.items()), frozenset({fact.fact_id})))
        return result
    if op == "filter":
        result = [row for row in rows if _predicate(row, params["where"])]
    elif op == "project":
        mapping = params["columns"]
        if not isinstance(mapping, dict) or not mapping:
            raise ContractError("missing_projection")
        result = [CellRow(tuple((name, row.get(field)) for name, field in mapping.items()), row.provenance)
                  for row in rows]
    elif op == "distinct":
        result = _deduplicate(rows, params["keys"])
    elif op == "calculate":
        result = []
        for row in rows:
            value = _combine_quantities(row.get(params["left"]), row.get(params["right"]), params["operator"])
            fields = dict(row.cells)
            fields[params["column"]] = value
            result.append(CellRow(tuple(fields.items()), row.provenance))
    elif op == "state_fold":
        states = {}
        for row in sorted(rows, key=lambda item: item.get("claim_order")):
            change = row.get("change")
            if change == "reference":
                continue
            if change not in {"present", "absent"}:
                raise ContractError("unresolved_membership_transition")
            key = _row_key(row, params["keys"])
            values = dict(row.cells)
            values["value"] = Quantity.from_raw(int(change == "present"), "1")
            old = states.get(key)
            states[key] = CellRow(tuple(values.items()), row.provenance | (old.provenance if old else frozenset()))
        result = list(states.values())
    elif op == "reduce":
        keys, aggregate = params["keys"], params["aggregate"]
        if not isinstance(keys, list) or aggregate not in {"sum", "mean", "count"}:
            raise ContractError("unsupported_reduction")
        groups = defaultdict(list)
        for row in rows:
            groups[_row_key(row, keys) if keys else ()].append(row)
        result = []
        for group in groups.values():
            if aggregate == "count":
                value = Quantity.from_raw(len(group), "1")
            else:
                values = [row.get(params["column"]) for row in group]
                value = values[0]
                if not isinstance(value, Quantity):
                    raise ContractError("reduction_requires_quantities")
                for item in values[1:]:
                    value = _combine_quantities(value, item, "add")
                if aggregate == "mean":
                    value = Quantity(value.amount / len(group), value.dimensions, value.display_unit)
            cells = [(key, group[0].get(key)) for key in keys] + [("value", value)]
            result.append(CellRow(tuple(cells), _provenance(group)))
    elif op in {"union", "intersection"}:
        left, right = inputs
        if op == "union":
            result = _deduplicate(left + right, params["keys"])
        else:
            right_by_key = {_row_key(row, params["keys"]): row for row in _deduplicate(right, params["keys"])}
            result = [CellRow(row.cells, row.provenance | right_by_key[_row_key(row, params["keys"])].provenance)
                      for row in _deduplicate(left, params["keys"])
                      if _row_key(row, params["keys"]) in right_by_key]
    elif op == "join":
        left, right = inputs
        pairs = params["on"]
        if not isinstance(pairs, list) or not pairs or not all(isinstance(pair, list) and len(pair) == 2 for pair in pairs):
            raise ContractError("missing_join_keys")
        cardinality = params["cardinality"]
        if cardinality not in {"one_to_one", "many_to_one", "many_to_many"}:
            raise ContractError("invalid_join_cardinality")
        left_keys, right_keys = [p[0] for p in pairs], [p[1] for p in pairs]
        counts = [collections_count(table, keys) for table, keys in ((left, left_keys), (right, right_keys))]
        if cardinality != "many_to_many" and any(n > 1 for n in counts[1].values()):
            raise ContractError("ambiguous_right_join_operand")
        if cardinality == "one_to_one" and any(n > 1 for n in counts[0].values()):
            raise ContractError("ambiguous_left_join_operand")
        result = []
        for a in left:
            for b in right:
                if _row_key(a, left_keys) == _row_key(b, right_keys):
                    if a.provenance == b.provenance and len(a.provenance) == 1 and step["inputs"][0] != step["inputs"][1]:
                        raise ContractError("independent_join_roles_reuse_one_claim")
                    values = [("left." + k, v) for k, v in a.cells] + [("right." + k, v) for k, v in b.cells]
                    result.append(CellRow(tuple(values), a.provenance | b.provenance))
                    if len(result) > MAX_ROWS:
                        raise ContractError("join_row_budget_exceeded")
    elif op == "arithmetic":
        left, right = inputs
        if len(left) != 1 or len(right) != 1:
            raise ContractError("arithmetic_requires_scalar_inputs")
        if (left[0].provenance == right[0].provenance
                and len(left[0].provenance) == 1 and step["inputs"][0] != step["inputs"][1]):
            raise ContractError("independent_operands_reuse_one_measurement_claim")
        value = _combine_quantities(left[0].get(params["left"]), right[0].get(params["right"]), params["operator"])
        result = [CellRow((("value", value),), _provenance(left + right))]
    elif op == "rank":
        if params["direction"] not in {"max", "min"}:
            raise ContractError("invalid_rank_direction")
        values = [row.get(params["column"]) for row in rows]
        if not all(isinstance(value, Quantity) for value in values):
            raise ContractError("rank_requires_quantities")
        if len({value.dimensions for value in values}) != 1:
            raise ContractError("incompatible_units")
        chooser = max if params["direction"] == "max" else min
        extreme = chooser(value.amount for value in values)
        # All tied rows survive. A scalar output must establish uniqueness.
        result = [row for row, value in zip(rows, values) if value.amount == extreme]
    else:
        raise ContractError("unsupported_operator")
    if not result:
        raise ContractError("empty_result_is_not_zero")
    if len(result) > MAX_ROWS:
        raise ContractError("row_budget_exceeded")
    return result


def collections_count(rows, keys):
    counts = defaultdict(int)
    for row in rows:
        counts[_row_key(row, keys)] += 1
    return counts


def _format_cell(value, target, digits):
    if not isinstance(value, Quantity):
        if target is not None:
            raise ContractError("unit_on_non_numeric_projection")
        return str(value)
    if target == "percent" or (target is None and value.display_unit == "percent"):
        if value.dimensions:
            raise ContractError("percent_requires_dimensionless_ratio")
        number, label = value.amount * 100, "%"
    else:
        label = target or value.display_unit
        if label == "canonical":
            raise ContractError("derived_unit_requires_output_unit")
        number = value.convert(label)
        label = unit(label).label
    if digits is not None:
        number = number.quantize(Decimal(1).scaleb(-digits))
    else:
        precision = max(4, 3 - number.adjusted()) if number else 4
        number = number.quantize(Decimal(1).scaleb(-precision)) if number.as_tuple().exponent < -precision else number
    text = format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")
    text = "0" if text in {"-0", ""} else text
    if label == "usd":
        return "$" + text
    if label == "%":
        return text + "%"
    if label == "1":
        return text
    if label in {"second", "minute", "hour", "day", "week", "month", "year", "item", "page"} and abs(number) != 1:
        label += "s"
    return text + " " + label


def execute_plan(plan_payload, facts, sources, *, aliases=None):
    """Execute supplied semantics. Structural validity is NOT semantic equivalence."""
    result = {"version": VERSION, "status": "NOT_PROVEN", "executable": False,
              "answer": None, "reason": None, "steps": [], "provenance": [],
              "scope": "supplied_source_view", "semantic_equivalence_verified": False}
    try:
        with localcontext() as context:
            context.prec = 40
            plan = validate_plan(plan_payload)
            facts = tuple(facts)
            if len(facts) > MAX_FACTS or not facts:
                raise ContractError("fact_budget_or_absence")
            unique = {}
            for fact in facts:
                if not isinstance(fact, Fact) or fact.source_id not in sources:
                    raise ContractError("unresolved_fact_provenance")
                if isinstance(fact, SelectedFact):
                    from PRAGMOS_grounded_computation import restore_selected_fact
                    reconstructed = restore_selected_fact(fact, sources)
                else:
                    reconstructed = ground_fact({
                        "source_id": fact.source_id, "quote": fact.quote, "start": fact.start,
                        "actor_span": fact.actor_span, "predicate": fact.predicate,
                        "status": fact.status, "kind": fact.kind, "operands": list(fact.operands),
                        "roles": dict(fact.roles), "measurement": (
                            {"text": fact.raw_measurement, "value": fact.raw_value, "unit": fact.raw_unit}
                            if fact.quantity else None
                        ),
                    }, sources, {o["id"] for o in plan["operands"]})
                if reconstructed != fact:
                    raise ContractError("fact_not_equal_to_grounded_source")
                old = unique.get(fact.fact_id)
                if old is not None and old != fact:
                    # Same source claim can belong to more than one operand.
                    view_fields = {"operands": (), "quote": "", "start": 0, "end": 0}
                    if {**old.__dict__, **view_fields} != {**fact.__dict__, **view_fields}:
                        raise ContractError("conflicting_duplicate_fact")
                    from dataclasses import replace
                    fact = replace(old, operands=tuple(sorted(set(old.operands) | set(fact.operands))))
                unique[fact.fact_id] = fact
            facts = tuple(unique.values())
            alias_lookup, alias_provenance = grounded_aliases(aliases, sources)
            values, operands = {}, {o["id"]: o for o in plan["operands"]}
            for step in plan["steps"]:
                rows = _execute_step(step, [values[key] for key in step["inputs"]], operands,
                                     facts, sources, alias_lookup)
                values[step["id"]] = rows
                result["steps"].append({
                    "id": step["id"], "op": step["op"], "row_count": len(rows),
                    "input_fact_ids": sorted(_provenance(rows)),
                    "rows": [{k: v.as_dict() if isinstance(v, Quantity) else v for k, v in row.cells} for row in rows],
                })
            output = plan["output"]
            rows = values[output["step"]]
            if output["shape"] == "scalar" and len(rows) != 1:
                raise ContractError("output_not_unique")
            rendered = [" | ".join(_format_cell(row.get(column), output["unit"], output["decimals"])
                                   for column in output["columns"]) for row in rows]
            provenance = _provenance(rows)
            result.update({"status": "COMPLETE", "executable": True,
                           "answer": ", ".join(rendered), "plan": plan,
                           "alias_bindings": alias_provenance,
                           "provenance": [
                               {**unique[key].as_dict(), "source": sources[unique[key].source_id].as_dict()}
                               for key in sorted(provenance)
                           ]})
    except (ContractError, InvalidOperation, TypeError, KeyError, ValueError) as exc:
        result.update({"status": "NOT_PROVEN", "executable": False, "answer": None,
                       "reason": str(exc)})
    return result
