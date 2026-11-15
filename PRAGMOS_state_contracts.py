"""Question-bound answer contracts and typed source relations for state views."""

from dataclasses import asdict, dataclass, replace

from PRAGMOS_composable_execution import ContractError, Quantity, normalized
from PRAGMOS_grounded_computation import choice, encoded, object_schema
from PRAGMOS_typed_claims import actor_choices, numeric_slots, span_choices


PROJECTIONS = ("attribute", "identity", "location", "duration", "frequency", "cardinality",
               "measurement", "boolean", "direction", "record", "collection", "unresolved")
DIMENSIONS = ("duration", "calendar_month", "calendar_year", "mass", "length", "currency",
              "count", "scalar", "none", "unresolved")
VALUE_TYPES = ("quantity", "text", "record")
FAMILIES = ("read", "reduce", "calculate", "compare", "predicate", "collect", "unresolved")
TIME_SELECTORS = ("latest", "earliest", "previous", "before_value", "all", "at_time", "unresolved")


def literal_schema(nullable=False):
    value = {"type": "string", "minLength": 1, "maxLength": 600}
    return {"anyOf": [{"type": "null"}, value]} if nullable else value


def require_literal(container, value, *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip() or value not in container:
        raise ContractError("state_contract_span_not_literal")


def require_fields(payload, schema):
    if not isinstance(payload, dict) or set(payload) != set(schema["properties"]):
        raise ContractError("state_answer_contract_invalid_fields")
    for field, definition in schema["properties"].items():
        if "enum" in definition and payload[field] not in definition["enum"]:
            raise ContractError("state_answer_contract_invalid_enum:" + field)


@dataclass(frozen=True)
class RequiredRelation:
    relation_id: str
    request_ref: str


@dataclass(frozen=True)
class AnswerContract:
    question: str
    request_ref: str
    subject_ref: str | None
    owner_scope: str
    target_ref: str | None
    relation_ref: str
    qualifiers: tuple
    projection: str
    input_type: str
    dimension: str
    counted_entity: str | None
    shape: str
    modality: str
    time_selector: str
    time_ref: str | None
    operation_family: str
    required_operands: tuple

    def as_dict(self):
        return {"version": 1, **asdict(self), "semantic_binding": "model_interpreted"}


def question_literals(question, *, nullable=False):
    spans = list(dict.fromkeys([question, *span_choices(question, width=len(question.split()) + 1)]))
    return choice([None, *spans] if nullable else spans)


def output_schema(question=None):
    schema = object_schema({"request_ref": literal_schema(), "projection": choice(PROJECTIONS),
                          "input_type": choice(VALUE_TYPES), "dimension": choice(DIMENSIONS),
                          "counted_entity": literal_schema(True),
                          "shape": choice(["one", "pair", "boolean", "list"]),
                          "operation_family": choice(FAMILIES)})
    if question is None:
        return schema
    # These are dependent type constraints, not classifications of question wording.
    types = {"duration": ("quantity", ["duration", "calendar_month", "calendar_year"]),
             "location": ("text", ["none"]), "identity": ("text", ["none"]),
             "record": ("record", ["none"]), "measurement": ("quantity", DIMENSIONS),
             "cardinality": ("quantity", ["count"]), "frequency": ("quantity", ["count"])}
    alternatives = []
    for projection in PROJECTIONS:
        fields = dict(schema["properties"], request_ref=question_literals(question), projection=choice([projection]),
                      counted_entity=question_literals(question, nullable=True))
        if projection in types:
            kind, dimensions = types[projection]
            fields.update(input_type=choice([kind]), dimension=choice(dimensions))
        if projection == "boolean":
            fields["shape"] = choice(["boolean"])
        if projection == "direction":
            fields["shape"] = choice(["one"])
        alternatives.append(object_schema(fields))
    return {"anyOf": alternatives}


def scope_schema(question=None):
    schema = object_schema({"subject_ref": literal_schema(True),
                          "owner_scope": choice(["speaker", "other", "unresolved"]),
                          "target_ref": literal_schema(True), "relation_ref": literal_schema(),
                          "qualifiers": {"type": "array", "items": literal_schema(), "maxItems": 8},
                          "modality": choice(["actual", "planned", "unresolved"]),
                          "time_selector": choice(TIME_SELECTORS), "time_ref": literal_schema(True),
                          "operand_refs": {"type": "array", "items": literal_schema(), "minItems": 1, "maxItems": 4}})
    if question is not None:
        fields = schema["properties"]
        fields["subject_ref"] = choice([None, *actor_choices(question)])
        for name in ("target_ref", "time_ref"):
            fields[name] = question_literals(question, nullable=True)
        fields["relation_ref"] = question_literals(question)
        fields["qualifiers"] = {**fields["qualifiers"], "items": question_literals(question)}
        fields["operand_refs"] = {**fields["operand_refs"], "items": question_literals(question)}
    return schema


def bind_answer_contract(question, output, scope):
    require_fields(output, output_schema())
    require_fields(scope, scope_schema())
    for field in ("request_ref", "counted_entity"):
        require_literal(question, output[field], nullable=field == "counted_entity")
    for field in ("subject_ref", "target_ref", "relation_ref", "time_ref"):
        require_literal(question, scope[field], nullable=field != "relation_ref")
    for field, maximum in (("qualifiers", 8), ("operand_refs", 4)):
        values = scope[field]
        if not isinstance(values, list) or len(values) > maximum or len(values) != len(set(values)):
            raise ContractError("state_answer_contract_invalid_spans:" + field)
        for value in values:
            require_literal(question, value)
    if not scope["operand_refs"]:
        raise ContractError("state_answer_contract_missing_operands")
    if "unresolved" in (output["projection"], output["dimension"], output["operation_family"],
                         scope["owner_scope"], scope["modality"]):
        raise ContractError("state_answer_contract_unresolved")
    projection, input_type, dimension = output["projection"], output["input_type"], output["dimension"]
    if projection == "duration" and (input_type != "quantity" or dimension not in {
            "duration", "calendar_month", "calendar_year"}):
        raise ContractError("state_duration_contract_requires_time_quantity")
    if projection in {"location", "identity"} and input_type != "text":
        raise ContractError("state_text_projection_requires_text_relation")
    if projection == "record" and input_type != "record":
        raise ContractError("state_record_projection_requires_structured_value")
    if projection in {"boolean", "direction"} and output["shape"] == "pair":
        raise ContractError("state_predicate_projection_cannot_discard_requested_pair")
    shape = "boolean" if projection == "boolean" else "one" if projection == "direction" else output["shape"]
    return AnswerContract(question, output["request_ref"], scope["subject_ref"], scope["owner_scope"],
                          scope["target_ref"], scope["relation_ref"], tuple(scope["qualifiers"]), projection,
                          input_type, dimension, output["counted_entity"], shape, scope["modality"],
                          scope["time_selector"], scope["time_ref"], output["operation_family"],
                          tuple(RequiredRelation("P" + str(i + 1), ref)
                                for i, ref in enumerate(scope["operand_refs"])))


def plan_answer_contract(ask, question):
    """Small question-only decisions form one shared contract."""
    suffix = "\nQuestion: " + encoded(question)
    projection = ask("Select the requested ANSWER TYPE. location=place; duration=elapsed time; frequency="
        "recurrence; cardinality=number of objects; measurement=another amount; record=compound numeric "
        "record such as wins/losses; identity=entity name; attribute=other text; boolean=Yes/No; direction="
        "increased/decreased; collection=list. A best achievement is not automatically a compound record. "
        "Return one type, without answering." + suffix,
        object_schema({"projection": choice(PROJECTIONS)}), 40, "planner")["projection"]
    family = ask("Select the required OPERATION. read retrieves a stated attribute, amount or duration; "
        "collect counts/lists distinct member identities; reduce selects min/max or aggregates separate "
        "measurements; calculate combines inputs arithmetically; compare compares two selected values; "
        "predicate establishes Yes/No. Count objects, not mentions of their changing numerical totals."
        + suffix, object_schema({"family": choice(FAMILIES)}), 40, "planner")["family"]
    shape = ("boolean" if projection == "boolean" else "one" if projection == "direction" else ask(
        "How many OUTPUTS are explicitly requested? one=one answer, pair=two or more separately requested "
        "values, list=a collection of members. Mentioning an old state or a condition does not itself request "
        "another output." + suffix, object_schema({"shape": choice(["one", "pair", "list"])}), 32, "planner")["shape"])
    input_type = ("record" if projection == "record" else "text" if family == "collect" or projection in {
        "location", "identity", "attribute", "collection"} else "quantity")
    if projection == "boolean":
        input_type = ask("Do the source operands needed by this question have type quantity, record or text? "
            "Do not classify the final Yes/No answer." + suffix,
            object_schema({"input_type": choice(VALUE_TYPES)}), 40, "planner")["input_type"]
    counted = (ask("Copy the noun phrase identifying WHAT is being counted, not the person, question words "
        "or an operation. Include distinguishing noun modifiers. Return null only if no counted noun is stated."
        + suffix, object_schema({"counted_entity": question_literals(question, nullable=True)}), 64, "planner")["counted_entity"]
        if projection == "cardinality" else None)
    owner = ask("Bind the requested HUMAN subject. subject_ref is its literal person reference, including "
        "a FULL relative phrase when present. owner_scope=speaker for the narrator, other for another person. "
        "A named person is not the narrator; a narrator's object/account is not another person." + suffix,
        object_schema({"subject_ref": choice([None, *actor_choices(question)]),
                       "owner_scope": choice(["speaker", "other", "unresolved"])}), 96, "planner")
    operands = ask("Copy each distinct SOURCE PROPERTY needed to answer. Different times of the same property "
        "use ONE operand, and comparing two people's same property also uses ONE operand with different owners. "
        "A ratio needs separate numerator and denominator properties. Use the whole distinguishing property "
        "phrase, not separate topic words." + suffix,
        object_schema({"operand_refs": {"type": "array", "items": question_literals(question), "minItems": 1, "maxItems": 4}}),
        160, "planner")["operand_refs"]
    scope = ask("Copy the requested object/account/event and mandatory restrictions from the question; "
        "do not split a composite entity or invent words. target_ref is null when implicit." + suffix,
        object_schema({"target_ref": question_literals(question, nullable=True),
            "qualifiers": {"type": "array", "items": question_literals(question), "maxItems": 8}}), 160, "planner")
    temporal = ask("Bind the requested time view. latest is the current/latest known observation, including "
        "the latest check-in. earliest is the original state; previous is the state before the latest change; "
        "before_value requires the literal value marker. at_time is an explicit dated/interval/event boundary. "
        "all is an aggregation over history. A vague past tense alone is latest. time_ref is null unless an "
        "explicit boundary/value must be selected. modality is actual or planned." + suffix,
        object_schema({"time_selector": choice(TIME_SELECTORS), "time_ref": question_literals(question, nullable=True),
                       "modality": choice(["actual", "planned", "unresolved"])}), 128, "planner")
    dimension = ("none" if input_type != "quantity" else "duration" if projection == "duration"
                 else "count" if projection in {"cardinality", "frequency"} else "scalar")
    return bind_answer_contract(question,
        {"request_ref": question, "projection": projection, "input_type": input_type, "dimension": dimension,
         "counted_entity": counted, "shape": shape, "operation_family": family},
        {**owner, **scope, **temporal, "relation_ref": question, "operand_refs": operands})


def bind_evidence_contract(ask, contract, facts, *, report=None):
    """Bind the request to source column types before executing any values."""
    schema = {o.relation_id: {"request": o.request_ref,
        "properties": sorted({f.attribute for f in facts if f.column_id == o.relation_id and f.attribute}),
        "types": sorted({f.relation.value.kind for f in facts if f.column_id == o.relation_id}),
        "dimensions": sorted({f.relation.value.dimension for f in facts if f.column_id == o.relation_id}),
        "owners": sorted({f.owner for f in facts if f.column_id == o.relation_id})}
        for o in contract.required_operands}
    if len(contract.required_operands) > 1:
        binding = ask("Are these requested operand phrases references to ONE shared property at different "
            "times/owners, or genuinely DISTINCT input properties? People, places, times and query verbs are "
            "not separate quantities. A ratio needs distinct numerator/denominator properties; two readings "
            "of the same property share one column. Do not calculate or drop a required dependency. Use "
            "unresolved if ambiguous.\nQuestion: " + encoded(contract.question)
            + "\nSource column schema (values withheld): " + encoded(schema),
            object_schema({"properties": choice(["shared", "distinct", "unresolved"])}), 64, "planner")["properties"]
        if binding == "unresolved":
            raise ContractError("state_property_identity_unresolved")
        if binding == "shared":
            if report is not None:
                report["property_bindings"] = {o.relation_id: "P1" for o in contract.required_operands}
            contract = replace(contract, required_operands=(RequiredRelation("P1", contract.question),))
    projection = ask("Bind the requested OUTPUT TYPE to the source column schema. A location is a place, "
        "identity an entity name, attribute other text, boolean Yes/No, direction increased/decreased. duration is time, "
        "frequency a recurrence, cardinality a count of objects, measurement another amount, record "
        "ONLY a compound record such as wins/losses (not the ordinary meaning of a best achievement). "
        "Do not compute or choose an answer. Use unresolved if source columns cannot support the request. "
        "\nQuestion: " + encoded(contract.question) + "\nColumn schema (values withheld): " + encoded(schema),
        object_schema({"projection": choice(PROJECTIONS)}), 48, "planner")["projection"]
    if projection == "unresolved":
        raise ContractError("state_source_answer_type_unresolved")
    kinds = {f.relation.value.kind for f in facts}
    if projection == "cardinality" and kinds == {"text"}:
        family, input_type = "collect", "text"
    else:
        options = (["read"] if projection in {"location", "identity", "attribute", "record"}
                   else ["compare"] if projection == "direction"
                   else ["compare", "predicate"] if projection == "boolean"
                   else ["read", "reduce", "calculate", "collect"])
        family = ask("Bind the requested operation FAMILY, without calculating any answers. read retrieves "
            "a stated value, including a cumulative total or elapsed duration; reduce compares or sums separate "
            "measurements; calculate combines distinct inputs; collect builds/counts member identities; compare "
            "compares two selected values; predicate establishes a proposition. Choose by the QUESTION and "
            "column types, not by how many mentions exist.\nQuestion: " + encoded(contract.question)
            + "\nOutput type: " + encoded(projection) + "\nColumn schema (values withheld): " + encoded(schema),
            object_schema({"family": choice(options)}), 48, "planner")["family"]
        input_type = next(iter(kinds)) if len(kinds) == 1 else contract.input_type
    dimensions = {f.relation.value.dimension for f in facts if f.relation.value.kind == input_type}
    dimension = next(iter(dimensions)) if len(dimensions) == 1 else contract.dimension
    shape = "boolean" if projection == "boolean" else "one" if projection in {"direction", "record"} else contract.shape
    if projection in {"cardinality", "duration", "measurement", "frequency"} and shape == "list":
        shape = "one"
    return replace(contract, projection=projection, operation_family=family, input_type=input_type, shape=shape,
                   dimension=dimension, counted_entity=contract.counted_entity if projection == "cardinality" else None)


def execution_contract(question, expression, groups):
    """One request contract follows the evidence-bound expression, never a classifier chain."""
    op = expression["op"]
    family = ({"read": "read", "minimum": "reduce", "maximum": "reduce", "sum": "reduce",
               "count": "collect", "members": "collect", "distinct": "collect", "exists": "predicate",
               "equal": "compare", "greater": "compare", "less": "compare", "direction": "compare"}.get(op, "calculate"))
    shape = "pair" if op == "tuple" else "boolean" if op in {"equal", "greater", "less", "exists"} else "list" if op in {"members", "distinct", "exclude"} else "one"
    if op == "tuple":
        families = {execution_contract(question, part, groups).operation_family for part in expression["inputs"]}
        family = families.pop() if len(families) == 1 else "compose"
    projection = ("boolean" if shape == "boolean" else "direction" if op == "direction" else
                  "collection" if shape == "list" else "attribute" if op == "read" else "measurement")
    def modalities(node):
        if "series" in node:
            yield node["modality"]
        for value in node.values():
            if isinstance(value, dict):
                yield from modalities(value)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        yield from modalities(child)
    statuses = set(modalities(expression))
    if len(statuses) != 1 or not statuses <= {"actual", "planned"}:
        raise ContractError("state_expression_mixed_or_unbound_modality")
    return AnswerContract(question, question, None, "evidence", None, question, (), projection, "text", "none",
        None, shape, statuses.pop(), "latest", None, family,
        tuple(RequiredRelation(row["id"], row["request"]) for row in groups))


@dataclass(frozen=True)
class TypedStateValue:
    raw: str
    kind: str
    dimension: str
    numbers: tuple = ()


def bind_typed_value(raw, kind, dimension, *, literal_rate=False):
    if kind not in VALUE_TYPES or dimension not in DIMENSIONS or dimension == "unresolved":
        raise ContractError("state_source_value_type_unresolved")
    slots = numeric_slots(raw)
    if kind == "quantity":
        if not slots and not literal_rate:
            raise ContractError("state_source_quantity_has_no_numeric_value")
        known = {name for slot in slots if slot.raw_unit
                 for name, _ in Quantity.from_raw(slot.value, slot.raw_unit).dimensions}
        if any(actual != dimension and not (actual.startswith("currency:") and dimension == "currency") for actual in known):
            raise ContractError("state_source_quantity_dimension_mismatch")
        if not known and dimension in {"duration", "calendar_month", "calendar_year", "mass", "length", "currency"}:
            raise ContractError("state_source_quantity_unit_not_bound")
    if kind == "record" and dimension != "none":
        raise ContractError("state_record_is_not_a_scalar_measurement")
    return TypedStateValue(raw, kind, dimension, tuple((slot.text, slot.value, slot.raw_unit) for slot in slots))


@dataclass(frozen=True)
class SourceStateRelation:
    operand_id: str | None
    subject_ref: str
    predicate_ref: str
    property_ref: str | None
    entity_ref: str | None
    value_role: str
    value: TypedStateValue
    assertion_ref: str
    qualifiers: tuple
    time_ref: str | None
    counted_entity_ref: str | None = None

    def history_key(self, owner):
        return (owner, self.operand_id, normalized(self.property_ref), normalized(self.entity_ref),
                self.value_role, self.value.kind, self.value.dimension)


def relation_matches_contract(relation, contract):
    operands = {operand.relation_id for operand in contract.required_operands}
    if relation.operand_id not in operands:
        return False
    if relation.value.kind != contract.input_type:
        raise ContractError("state_relation_input_type_mismatch")
    if contract.projection in {"location", "identity", "duration", "frequency", "record"}:
        if relation.value_role != contract.projection:
            raise ContractError("state_relation_answer_role_mismatch")
    if contract.dimension not in {"none", "scalar"} and relation.value.dimension != contract.dimension:
        raise ContractError("state_relation_answer_dimension_mismatch")
    if contract.counted_entity and relation.value.dimension == "count":
        from PRAGMOS_typed_claims import count_dimension
        if not relation.counted_entity_ref or count_dimension(relation.counted_entity_ref) != count_dimension(contract.counted_entity):
            raise ContractError("state_relation_counted_entity_requires_a_binding")
    return True


def validate_relation_histories(facts, *, modality=None):
    eligible = [fact for fact in facts if fact.kind not in {"reference", "context"}
                and (modality is None or fact.modality == modality)]
    if modality is None:
        for status in {fact.modality for fact in eligible}:
            validate_relation_histories(eligible, modality=status)
        return
    bound = [fact for fact in eligible if fact.relation is not None]
    interpreted = [fact for fact in bound if (fact.claim_review or {}).get("method") == "independent_assertion_binding"]
    if interpreted:
        if (len(interpreted) != len(eligible) or len({(f.owner, f.column_id) for f in interpreted}) != 1
                or any((f.claim_review or {}).get("history_binding") != f.column_id
                       or (f.claim_review or {}).get("target_match") != "bound"
                       or (f.claim_review or {}).get("qualification_match") != "bound" for f in interpreted)
                ):
            raise ContractError("state_operand_crosses_bound_properties_or_owners")
        return
    if bound and (len(bound) != len(eligible) or len({fact.relation.history_key(fact.owner) for fact in bound}) != 1):
        raise ContractError("state_operand_crosses_properties")
    predicates = {normalized(fact.relation.predicate_ref) for fact in bound}
    if len(predicates) > 1:
        edges = {predicate: set() for predicate in predicates}
        for newer in bound:
            if not newer.previous or not newer.transition_quote:
                continue
            if newer.previous not in newer.transition_quote or newer.value not in newer.transition_quote:
                raise ContractError("state_relation_transition_not_witnessed")
            for older in bound:
                if older.fact_id != newer.fact_id and normalized(older.value) == normalized(newer.previous):
                    a, b = normalized(older.relation.predicate_ref), normalized(newer.relation.predicate_ref)
                    edges[a].add(b)
                    edges[b].add(a)
        reached, pending = set(), [next(iter(predicates))]
        while pending:
            predicate = pending.pop()
            if predicate not in reached:
                reached.add(predicate)
                pending.extend(edges[predicate] - reached)
        if reached != predicates:
            raise ContractError("state_operand_crosses_properties:source_predicates_require_binding")
