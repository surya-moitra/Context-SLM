#!/usr/bin/env python3

"""Typed evidence compilation for PRAGMOS temporal and multi-session queries.

This module is intentionally independent of LongMemEval labels and answers. It
turns a question into a small universal operation plan, retains raw evidence as
provenance-backed atoms, and binds operands to evidence before deterministic
execution. Callers decide whether a validated proof is allowed to replace their
legacy answer path.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import re
from collections import defaultdict
from typing import Any, Callable, Iterable


ALLOWED_OPERATORS = {
    "LOOKUP",
    "DATE_LOOKUP",
    "COUNT",
    "COUNT_DISTINCT",
    "SUM",
    "AVERAGE",
    "DIFFERENCE",
    "ELAPSED_TIME",
    "RELATIVE_JOIN",
    "ORDER",
    "LATEST",
    "EARLIEST",
    "COLLECT",
    "UNION",
    "INTERSECTION",
    "COMPARE",
    "ARGMAX",
    "ARGMIN",
}
MULTI_OPERAND_OPERATORS = {
    "DIFFERENCE",
    "ELAPSED_TIME",
    "ORDER",
    "COMPARE",
    "INTERSECTION",
}
COLLECTION_OPERATORS = {
    "COUNT", "COUNT_DISTINCT", "COLLECT", "UNION", "AVERAGE",
    "ARGMAX", "ARGMIN",
}
EVIDENCE_OUTCOMES = {
    "COMPLETE",
    "PARTIAL",
    "CONFLICTING",
    "EXTRACTION_FAILED",
    "ABSENT",
}
_SEMANTIC_PLAN_GRAMMARS = {}
_CANDIDATE_ADJUDICATION_GRAMMAR = None
_EVIDENCE_FRAME_GRAMMAR = None

ANSWER_ROLES = {
    "amount", "count", "date", "duration", "instrument", "label",
    "location", "object", "text", "theme", "tuple", "unknown",
}
GROUP_ROLES = {
    "event", "identity", "instrument", "label", "location", "object",
    "theme", "unknown",
}
REQUESTED_STATUSES = {"any", "completed", "planned", "unknown"}
RESULT_SHAPES = {"scalar", "tuple_set", "unknown", "value_set"}
PROJECTION_ROLES = {
    "amount", "count", "date", "duration", "event", "identity",
    "instrument", "label", "location", "object", "text", "theme",
}


# This is a small operation ontology, not a benchmark-question parser. It is
# shared by query validation and evidence compilation so that a count of events
# cannot silently become a sum of unrelated numbers.
EVENT_FAMILIES = {
    "acquisition": {
        "acquire", "acquired", "acquisition", "buy", "bought", "obtain",
        "obtained", "purchase", "purchased", "receive", "received",
    },
    "attendance": {"attend", "attended", "join", "joined"},
    "completion": {"complete", "completed", "finish", "finished"},
    "communication": {"call", "called", "email", "emailed", "message", "messaged"},
    "travel": {"travel", "traveled", "travelled", "trip"},
    "visit": {"visit", "visited"},
}
COMPLETED_EVENT_PREDICATES = {
    ("acquired",): "acquisition",
    ("attended",): "attendance",
    ("bought",): "acquisition",
    ("brought", "home"): "acquisition",
    ("called",): "communication",
    ("completed",): "completion",
    ("emailed",): "communication",
    ("finished",): "completion",
    ("joined",): "attendance",
    ("messaged",): "communication",
    ("obtained",): "acquisition",
    ("picked", "up"): "acquisition",
    ("purchased",): "acquisition",
    ("received",): "acquisition",
    ("traveled",): "travel",
    ("travelled",): "travel",
    ("visited",): "visit",
}
PENDING_EVENT_PREDICATES = {
    ("acquire",): "acquisition",
    ("attend",): "attendance",
    ("bring", "home"): "acquisition",
    ("buy",): "acquisition",
    ("call",): "communication",
    ("complete",): "completion",
    ("email",): "communication",
    ("finish",): "completion",
    ("join",): "attendance",
    ("message",): "communication",
    ("obtain",): "acquisition",
    ("pick", "up"): "acquisition",
    ("purchase",): "acquisition",
    ("receive",): "acquisition",
    ("travel",): "travel",
    ("visit",): "visit",
}
EVENT_OPERATION_TOKENS = {
    "acquisition", "acquire", "acquired", "bought", "buy", "completed",
    "complete", "count", "distinct", "event", "events", "item", "items",
    "number", "obtained", "purchase", "purchased", "recorded", "separate",
    "setup", "thing", "things", "total",
}
GENERIC_EVENT_OBJECTS = {
    "acquisition", "event", "item", "items", "purchase", "setup", "thing",
    "things",
}

# Contract words describe how to operate on evidence, not what the evidence is
# about.  Removing them before scope comparison prevents a generic word such as
# "charge" or "recorded" from admitting an otherwise unrelated measurement.
SEMANTIC_CONTRACT_TOKENS = {
    "acquire", "acquired", "acquisition", "activity", "amount", "arithmetic",
    "average", "charge", "combined", "completed", "count",
    "different", "distinct", "event", "finished", "item", "items", "list",
    "length", "location", "locations", "material", "materials", "mean",
    "number", "object", "objects", "obtained", "paired", "pair", "place",
    "places", "purchase", "purchased", "record",
    "recorded", "records", "result", "separate", "set", "sum", "thing",
    "things", "total", "tool", "tools", "verified", "many", "make",
}
MAPPING_CONTRACT_TOKENS = {
    "associate", "associated", "belong", "belongs", "corresponding", "each",
    "map", "mapping", "pair", "paired",
}
GENERIC_ROLE_TOKENS = {
    "activity", "acquisition", "context", "event", "item", "items",
    "location", "locations", "object", "objects", "project", "purchase",
    "record", "records", "setup", "thing", "things",
}
COUNT_SCOPE_CONTROL_WORDS = {
    "actual", "actually", "across", "after", "all", "altogether", "also",
    "cardinality", "check", "clarity", "collect", "collection", "confirmed",
    "conversation", "counting", "deduplicating", "duplicates", "each",
    "every", "history", "individual", "mention", "mentioned", "once",
    "only", "own", "precise", "recorded", "reflection", "remain",
    "reminder", "revisiting", "session", "supported", "tally", "those",
    "timeline", "use", "used", "using",
}


def semantic_plan_json_schema(
    include_result_contract: bool = True,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "operator": {"type": "string", "enum": sorted(ALLOWED_OPERATORS)},
            "answer_type": {"type": "string"},
            "target_unit": {"type": ["string", "null"]},
            "actor": {"type": ["string", "null"]},
            "answer_role": {
                "type": "string",
                "enum": sorted(ANSWER_ROLES),
            },
            "group_role": {
                "type": "string",
                "enum": sorted(GROUP_ROLES),
            },
            "requested_status": {
                "type": "string",
                "enum": sorted(REQUESTED_STATUSES),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "operands": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["event", "quantity", "item", "fact"],
                        },
                        "actor": {"type": ["string", "null"]},
                        "hard_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "id",
                        "text",
                        "kind",
                        "actor",
                        "hard_entities",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "operator",
            "answer_type",
            "target_unit",
            "actor",
            "answer_role",
            "group_role",
            "requested_status",
            "confidence",
            "operands",
        ],
        "additionalProperties": False,
    }
    if include_result_contract:
        schema["properties"].update(
            {
                "result_shape": {
                    "type": "string",
                    "enum": sorted(RESULT_SHAPES),
                },
                "projection_roles": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "string",
                        "enum": sorted(PROJECTION_ROLES),
                    },
                },
            }
        )
        schema["required"][7:7] = ["result_shape", "projection_roles"]
    return schema


def semantic_plan_grammar(include_result_contract: bool = True):
    """Build the local llama.cpp JSON grammar lazily and cache it."""
    key = bool(include_result_contract)
    if _SEMANTIC_PLAN_GRAMMARS.get(key) is False:
        return None
    if key not in _SEMANTIC_PLAN_GRAMMARS:
        try:
            from llama_cpp import LlamaGrammar

            _SEMANTIC_PLAN_GRAMMARS[key] = LlamaGrammar.from_json_schema(
                json.dumps(semantic_plan_json_schema(include_result_contract))
            )
        except Exception:
            _SEMANTIC_PLAN_GRAMMARS[key] = False
    return _SEMANTIC_PLAN_GRAMMARS[key] or None


def candidate_adjudication_json_schema() -> dict[str, Any]:
    match_values = ["yes", "no", "unknown", "not_applicable"]
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "include": {"type": "boolean"},
                        "actor_match": {"type": "string", "enum": match_values},
                        "event_match": {"type": "string", "enum": match_values},
                        "time_match": {"type": "string", "enum": match_values},
                        "scope_match": {"type": "string", "enum": match_values},
                        "value_grounded": {"type": "string", "enum": match_values},
                        "scope_basis": {
                            "type": "string",
                            "enum": [
                                "quote",
                                "stable_taxonomy",
                                "not_applicable",
                                "unknown",
                            ],
                        },
                        "canonical_identity": {"type": ["string", "null"]},
                        "item_quantity": {"type": ["number", "null"]},
                        "status": {
                            "type": "string",
                            "enum": ["completed", "pending", "planned", "negated", "unknown"],
                        },
                    },
                    "required": [
                        "candidate_id", "include", "actor_match", "event_match",
                        "time_match", "scope_match", "value_grounded",
                        "scope_basis", "canonical_identity", "item_quantity",
                        "status",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def candidate_adjudication_grammar():
    global _CANDIDATE_ADJUDICATION_GRAMMAR
    if _CANDIDATE_ADJUDICATION_GRAMMAR is False:
        return None
    if _CANDIDATE_ADJUDICATION_GRAMMAR is None:
        try:
            from llama_cpp import LlamaGrammar

            _CANDIDATE_ADJUDICATION_GRAMMAR = LlamaGrammar.from_json_schema(
                json.dumps(candidate_adjudication_json_schema())
            )
        except Exception:
            _CANDIDATE_ADJUDICATION_GRAMMAR = False
    return _CANDIDATE_ADJUDICATION_GRAMMAR or None


def evidence_frame_json_schema() -> dict[str, Any]:
    nullable_span = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "evidence_id": {"type": "string"},
                        "relevant": {"type": "boolean"},
                        "actor_reference": {
                            "type": "string",
                            "enum": ["source_speaker", "other", "unknown"],
                        },
                        "status": {
                            "type": "string",
                            "enum": ["completed", "planned", "negated", "unknown"],
                        },
                        "predicate": nullable_span,
                        "subject": nullable_span,
                        "object": nullable_span,
                        "instrument": nullable_span,
                        "theme": nullable_span,
                        "location": nullable_span,
                        "label": nullable_span,
                        "event_identity": nullable_span,
                        "event_context": nullable_span,
                        "value": {"type": ["number", "null"]},
                        "unit": nullable_span,
                        "quantity": {"type": "number"},
                    },
                    "required": [
                        "evidence_id", "relevant", "actor_reference", "status",
                        "predicate", "subject", "object", "instrument", "theme",
                        "location", "label", "event_identity",
                        "event_context", "value", "unit", "quantity",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["frames"],
        "additionalProperties": False,
    }


def evidence_frame_grammar():
    global _EVIDENCE_FRAME_GRAMMAR
    if _EVIDENCE_FRAME_GRAMMAR is False:
        return None
    if _EVIDENCE_FRAME_GRAMMAR is None:
        try:
            from llama_cpp import LlamaGrammar

            _EVIDENCE_FRAME_GRAMMAR = LlamaGrammar.from_json_schema(
                json.dumps(evidence_frame_json_schema())
            )
        except Exception:
            _EVIDENCE_FRAME_GRAMMAR = False
    return _EVIDENCE_FRAME_GRAMMAR or None


@dataclasses.dataclass(frozen=True)
class QueryOperand:
    operand_id: str
    text: str
    kind: str = "fact"
    actor: str | None = None
    hard_entities: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SemanticQueryPlan:
    operator: str
    answer_type: str
    target_unit: str | None
    actor: str | None
    operands: tuple[QueryOperand, ...]
    confidence: str
    status: str
    reason: str | None = None
    raw_output: str = ""
    source: str = "local_slm_json"
    answer_role: str = "unknown"
    group_role: str = "unknown"
    requested_status: str = "unknown"
    result_shape: str = "unknown"
    projection_roles: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    @property
    def trusted_for_legacy_routing(self) -> bool:
        return self.valid and self.confidence == "high"

    @property
    def requires_diverse_evidence(self) -> bool:
        return self.operator in (
            MULTI_OPERAND_OPERATORS
            | COLLECTION_OPERATORS
            | {"EARLIEST", "LATEST", "RELATIVE_JOIN", "SUM"}
        )

    @property
    def intent_family(self) -> str:
        """Return a benchmark-independent semantic family for routing/auditing."""
        if self.operator in {
            "DATE_LOOKUP", "ELAPSED_TIME", "RELATIVE_JOIN", "ORDER",
            "LATEST", "EARLIEST",
        }:
            return "temporal"
        if self.operator == "LOOKUP" and (
            self.answer_role in {"date", "duration"}
            or self.answer_type in {
                "date", "datetime", "day", "duration", "elapsed time",
                "month", "time", "week", "year",
            }
        ):
            return "temporal"
        if self.operator in {
            "COUNT", "COUNT_DISTINCT", "SUM", "AVERAGE", "DIFFERENCE",
            "COMPARE", "ARGMAX", "ARGMIN",
        }:
            return "aggregate"
        if self.operator in {"COLLECT", "UNION", "INTERSECTION"}:
            return "collection"
        return "factual"

    @property
    def temporal_operation(self) -> str:
        if self.intent_family != "temporal":
            return "none"
        return {
            "DATE_LOOKUP": "date_lookup",
            "ELAPSED_TIME": "elapsed_time",
            "RELATIVE_JOIN": "relative_join",
            "ORDER": "order",
            "LATEST": "latest",
            "EARLIEST": "earliest",
            "LOOKUP": "date_lookup",
        }.get(self.operator, "none")

    def retrieval_queries(self, question: str) -> list[str]:
        queries = []
        for operand in self.operands:
            parts = [operand.actor or self.actor, operand.text]
            if self.operator in {"COUNT", "COUNT_DISTINCT"} and self.group_role == "instrument":
                scope = count_scope_contract(self, question)
                if scope["topic_source"] == "question" and scope["topic_tokens"]:
                    parts = [
                        operand.actor or self.actor,
                        " ".join(scope["topic_tokens"]), operand.text,
                    ]
            query = " ".join(part for part in parts if part).strip()
            if query and query.lower() != question.strip().lower():
                queries.append(query)
        return list(dict.fromkeys(queries))

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["valid"] = self.valid
        payload["trusted_for_legacy_routing"] = self.trusted_for_legacy_routing
        payload["requires_diverse_evidence"] = self.requires_diverse_evidence
        payload["intent_family"] = self.intent_family
        payload["temporal_operation"] = self.temporal_operation
        return payload


def validate_semantic_route(
    plan: SemanticQueryPlan,
    question: str,
    *,
    question_timestamp: str | None = None,
) -> dict[str, Any]:
    """Validate a planner route without benchmark labels or answer metadata.

    The local model supplies semantics, but it does not get unilateral control
    over routing. Structural and grounding checks must pass first; a complete
    evidence proof is still required later before a compiled answer may replace
    the established generation path.
    """
    checks = {
        "schema_valid": isinstance(plan, SemanticQueryPlan) and plan.valid,
        "operator_supported": False,
        "operand_cardinality_valid": False,
        "operands_question_grounded": False,
        "answer_contract_valid": False,
        "reference_time_available": True,
    }
    result = {
        "schema_version": 1,
        "eligible": False,
        "intent_family": "unknown",
        "temporal_operation": "none",
        "checks": checks,
        "reason": "invalid_semantic_plan",
        "policy": (
            "label_blind_semantic_plan_plus_structural_validation; compiled "
            "answers additionally require a complete provenance-backed proof"
        ),
    }
    if not checks["schema_valid"]:
        return result

    result["intent_family"] = plan.intent_family
    result["temporal_operation"] = plan.temporal_operation
    checks["operator_supported"] = plan.operator in ALLOWED_OPERATORS

    operand_count = len(plan.operands)
    if plan.operator == "ELAPSED_TIME":
        checks["operand_cardinality_valid"] = operand_count >= 1
    elif plan.operator in MULTI_OPERAND_OPERATORS:
        checks["operand_cardinality_valid"] = operand_count >= 2
    else:
        checks["operand_cardinality_valid"] = operand_count >= 1

    question_tokens = content_tokens(question)
    checks["operands_question_grounded"] = bool(question_tokens) and all(
        bool(content_tokens(operand.text).intersection(question_tokens))
        and all(
            normalize_text(entity) in normalize_text(question)
            for entity in operand.hard_entities
        )
        for operand in plan.operands
    )

    if plan.operator in {"COUNT", "COUNT_DISTINCT"}:
        checks["answer_contract_valid"] = plan.answer_role in {"count", "unknown"}
    elif plan.operator in {"SUM", "AVERAGE", "DIFFERENCE", "COMPARE"}:
        checks["answer_contract_valid"] = plan.answer_role in {
            "amount", "count", "duration", "unknown",
        }
    elif plan.operator in {"ELAPSED_TIME"}:
        checks["answer_contract_valid"] = plan.answer_role in {
            "duration", "unknown",
        }
    elif plan.operator == "DATE_LOOKUP":
        checks["answer_contract_valid"] = plan.answer_role in {"date", "unknown"}
    elif plan.operator == "RELATIVE_JOIN":
        checks["answer_contract_valid"] = plan.answer_role in ANSWER_ROLES
    elif plan.operator == "LOOKUP" and plan.intent_family == "temporal":
        checks["answer_contract_valid"] = plan.answer_role in {"date", "duration"}
    else:
        checks["answer_contract_valid"] = True

    if plan.operator == "ELAPSED_TIME" and operand_count == 1:
        checks["reference_time_available"] = bool(question_timestamp)

    # Factual LOOKUP remains on the established answer path. It is a valid
    # plan for auditing, but does not justify invoking a deterministic compiler.
    route_has_compiler_work = not (
        plan.operator == "LOOKUP" and plan.intent_family == "factual"
    )
    result["eligible"] = route_has_compiler_work and all(checks.values())
    if result["eligible"]:
        result["reason"] = "validated_semantic_route"
    elif not route_has_compiler_work:
        result["reason"] = "factual_lookup_uses_existing_answer_path"
    else:
        result["reason"] = "semantic_route_failed_structural_validation"
    return result


@dataclasses.dataclass(frozen=True)
class EvidenceAtom:
    atom_id: str
    source_turn_id: str | None
    source_session_id: str | None
    source_role: str | None
    source_speaker: str | None
    source_timestamp: str | None
    source_quote: str
    clause: str
    clause_index: int
    tokens: tuple[str, ...]
    temporal_expression: dict[str, Any] | None = None
    event_status: str | None = None

    @property
    def provenance_identity(self) -> tuple[Any, ...]:
        return (
            self.source_session_id,
            self.source_turn_id,
            self.clause_index,
            normalize_text(self.clause),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["provenance_identity"] = list(self.provenance_identity)
        return payload


@dataclasses.dataclass(frozen=True)
class EventFact:
    event_id: str
    actor: str
    family: str
    predicate: str
    object_identity: str
    quantity: float
    status: str
    topic_coverage: float
    source_turn_id: str | None
    source_session_id: str | None
    source_role: str | None
    source_speaker: str | None
    source_quote: str
    clause: str
    provenance_identity: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["provenance_identity"] = list(self.provenance_identity)
        return payload


@dataclasses.dataclass(frozen=True)
class EvidenceFrame:
    frame_id: str
    evidence_id: str
    relevant: bool
    actor_reference: str
    status: str
    predicate: str | None
    subject: str | None
    object: str | None
    instrument: str | None
    theme: str | None
    location: str | None
    label: str | None
    event_identity: str | None
    event_context: str | None
    value: float | None
    unit: str | None
    quantity: float
    source_turn_id: str | None
    source_session_id: str | None
    source_role: str | None
    source_speaker: str | None
    source_timestamp: str | None
    source_quote: str
    clause: str
    provenance_identity: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["provenance_identity"] = list(self.provenance_identity)
        return payload


@dataclasses.dataclass(frozen=True)
class BoundFact:
    operand_id: str
    atom_id: str
    score: float
    token_coverage: float
    actor_match: bool
    hard_entity_match: bool
    provenance_identity: tuple[Any, ...]
    temporal_expression: dict[str, Any] | None
    source_turn_id: str | None
    source_session_id: str | None
    source_quote: str
    clause: str

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["provenance_identity"] = list(self.provenance_identity)
        return payload


@dataclasses.dataclass(frozen=True)
class OperationProof:
    status: str
    operator: str
    answer: str | None
    bindings: tuple[BoundFact, ...] = ()
    calculation: dict[str, Any] | None = None
    reason: str | None = None
    executable: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["bindings"] = [binding.as_dict() for binding in self.bindings]
        return payload


def prefer_stricter_more_complete_proof(
    current: OperationProof | None,
    candidate: OperationProof | None,
) -> OperationProof | None:
    """Prefer broader evidence only when the candidate carries a strict proof.

    This is deliberately monotonic: a validated existing proof is retained unless
    the replacement covers more provenance or sessions and every certificate
    condition is true. It prevents a larger but weakly filtered candidate set
    from displacing a known-good deterministic answer.
    """
    if candidate is None or not candidate.executable or candidate.status != "COMPLETE":
        return current
    if current is None or not current.executable or current.status != "COMPLETE":
        return candidate
    candidate_calculation = candidate.calculation or {}
    certificate = candidate_calculation.get("proof_certificate") or {}
    if not certificate or not all(certificate.values()):
        return current

    def coverage(proof: OperationProof) -> tuple[int, int]:
        calculation = proof.calculation or {}
        sessions = int(calculation.get("covered_session_count") or 0)
        provenance = calculation.get("provenance") or []
        return sessions, len(provenance)

    return candidate if coverage(candidate) > coverage(current) else current


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def normalize_phrase(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value)).strip()


def content_tokens(value: Any) -> set[str]:
    ignored = {
        "a", "an", "and", "are", "as", "at", "be", "before", "between",
        "by", "did", "do", "does", "for", "from", "had", "has", "have",
        "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
        "our", "the", "this", "to", "was", "were", "what", "when", "which",
        "who", "with", "you", "your",
    }
    irregular = {
        "bought": "buy", "did": "do", "drove": "drive", "gave": "give",
        "gone": "go", "made": "make", "ran": "run", "spent": "spend",
        "went": "go", "wore": "wear",
    }
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", normalize_text(value)):
        if token in ignored or len(token) <= 2:
            continue
        token = irregular.get(token, token)
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def effective_semantic_operator(plan: SemanticQueryPlan) -> str:
    """Type-check a planner operator without inspecting benchmark labels.

    A SUM requires additive measurements. A unitless set of item/event
    operands is a cardinality request even if a small model emitted SUM.
    """
    if not isinstance(plan, SemanticQueryPlan) or not plan.valid:
        return plan.operator if isinstance(plan, SemanticQueryPlan) else "LOOKUP"
    if plan.result_shape == "tuple_set" and plan.operator in {
        "INTERSECTION", "LOOKUP", "UNION",
    }:
        return "COLLECT"
    if plan.result_shape == "value_set" and plan.operator == "LOOKUP":
        return "COLLECT"
    if plan.operator != "SUM":
        return plan.operator
    if plan.answer_role in {"amount", "duration"}:
        return plan.operator
    operand_kinds = {normalize_text(operand.kind) for operand in plan.operands}
    target_unit = normalize_text(plan.target_unit)
    if operand_kinds and operand_kinds.issubset({"event", "item"}) and target_unit in {
        "", "item", "items", "number",
    }:
        return "COUNT"
    return plan.operator


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def semantic_plan_prompt(
    question: str,
    include_result_contract: bool = True,
) -> str:
    operators = ", ".join(sorted(ALLOWED_OPERATORS))
    result_contract = ""
    schema_contract = ""
    if include_result_contract:
        result_contract = (
            "result_shape is scalar for one number/value, value_set for a list "
            "of one semantic role, and tuple_set for paired or multi-column "
            "rows. projection_roles lists the exact roles required in each "
            "answer row; for example [object, location]. INTERSECTION is only "
            "for values shared by two real sets; use COLLECT for one set of "
            "values or tuples. "
        )
        schema_contract = (
            '"result_shape":"scalar|value_set|tuple_set",'
            '"projection_roles":[],'
        )
    return (
        "<|user|>\nYou are a semantic query planner. Return JSON only. Do not "
        "answer the question and do not use outside knowledge. Select one universal "
        f"operator from: {operators}. COUNT counts eligible events or items; "
        "COUNT_DISTINCT counts unique identities; SUM adds stated quantities; "
        "AVERAGE computes their arithmetic mean; DIFFERENCE subtracts two bound "
        "quantities; ARGMAX or ARGMIN returns the associated label, not the "
        "extreme number, when a label is requested; ELAPSED_TIME compares event "
        "timestamps; DATE_LOOKUP returns the time of one event; RELATIVE_JOIN "
        "finds an event or date at a stated offset from another event. Identify "
        "the facts or events that must be "
        "grounded before answering. hard_entities must contain only exact named "
        "entities or distinctive noun phrases copied from the question; do not put "
        "generic words there. answer_role is the requested output type. group_role "
        "is the semantic role whose distinct values are counted, collected, or "
        "returned; for example an object, instrument, label, location, or event. "
        "requested_status is completed, planned, any, or unknown. "
        f"{result_contract}Use actor=user "
        "for first-person memories. confidence must be high, medium, or low. Schema:\n"
        '{"operator":"...","answer_type":"...","target_unit":null,'
        '"actor":null,"answer_role":"...","group_role":"...",'
        '"requested_status":"...",'
        f'{schema_contract}'
        '"confidence":"...","operands":[{"id":"O1",'
        '"text":"...","kind":"event|quantity|item|fact","actor":null,'
        '"hard_entities":[]}]}\n'
        f"Question: {question}\n<|end|>\n<|assistant|>\n"
    )


def _invalid_plan(reason: str, raw_output: str = "") -> SemanticQueryPlan:
    return SemanticQueryPlan(
        operator="LOOKUP",
        answer_type="unknown",
        target_unit=None,
        actor=None,
        operands=(),
        confidence="low",
        status="invalid",
        reason=reason,
        raw_output=raw_output,
    )


def parse_semantic_plan(payload: Any, question: str, raw_output: str = "") -> SemanticQueryPlan:
    if not isinstance(payload, dict):
        return _invalid_plan("planner_output_not_json_object", raw_output)
    operator = str(payload.get("operator") or "").strip().upper()
    if operator not in ALLOWED_OPERATORS:
        return _invalid_plan("operator_not_allowed", raw_output)
    confidence = normalize_text(payload.get("confidence"))
    if confidence not in {"high", "medium", "low"}:
        return _invalid_plan("invalid_confidence", raw_output)
    raw_operands = payload.get("operands")
    if not isinstance(raw_operands, list) or not raw_operands:
        return _invalid_plan("operands_missing", raw_output)

    normalized_question = normalize_text(question)
    inferred_actor = str(payload.get("actor") or "").strip() or None
    if inferred_actor is None and re.search(
        r"\b(?:i|me|my|mine|we|our|ours)\b",
        question,
        flags=re.IGNORECASE,
    ):
        inferred_actor = "user"
    operands = []
    warnings = []
    seen_ids = set()
    for index, value in enumerate(raw_operands[:8], start=1):
        if not isinstance(value, dict):
            return _invalid_plan("operand_not_object", raw_output)
        operand_id = str(value.get("id") or f"O{index}").strip().upper()
        text = re.sub(r"\s+", " ", str(value.get("text") or "")).strip()
        if not text or operand_id in seen_ids or not content_tokens(text):
            return _invalid_plan("operand_invalid_or_duplicate", raw_output)
        seen_ids.add(operand_id)
        hard_entities = []
        for entity in value.get("hard_entities") or []:
            entity = re.sub(r"\s+", " ", str(entity)).strip()
            if not entity:
                continue
            if normalize_text(entity) not in normalized_question:
                warnings.append(f"dropped_ungrounded_hard_entity:{entity}")
                continue
            hard_entities.append(entity)
        operands.append(
            QueryOperand(
                operand_id=operand_id,
                text=text,
                kind=normalize_text(value.get("kind") or "fact"),
                actor=str(value.get("actor") or "").strip() or inferred_actor,
                hard_entities=tuple(dict.fromkeys(hard_entities)),
            )
        )

    answer_role = normalize_text(payload.get("answer_role") or "")
    if answer_role not in ANSWER_ROLES:
        answer_type = normalize_text(payload.get("answer_type") or "unknown")
        answer_role = answer_type if answer_type in ANSWER_ROLES else "unknown"
    group_role = normalize_text(payload.get("group_role") or "unknown")
    if group_role not in GROUP_ROLES:
        group_role = "unknown"
    if operator in {"COUNT", "COUNT_DISTINCT"}:
        # A planner may put the entity being counted in answer_role (for
        # example, instrument).  The result is still a count; preserve that
        # entity role as the grouping contract instead of rejecting the route.
        if answer_role in GROUP_ROLES - {"unknown"} and group_role in {
            "event", "unknown",
        }:
            group_role = answer_role
            warnings.append("moved_counted_entity_role_to_group_role")
        if answer_role != "count":
            answer_role = "count"
            warnings.append("normalized_count_answer_role")
    requested_status = normalize_text(payload.get("requested_status") or "unknown")
    if requested_status not in REQUESTED_STATUSES:
        requested_status = "unknown"
    result_shape = normalize_text(payload.get("result_shape") or "")
    if result_shape not in RESULT_SHAPES:
        if answer_role == "tuple":
            result_shape = "tuple_set"
        elif operator in {"COLLECT", "UNION", "INTERSECTION"}:
            result_shape = "value_set"
        else:
            result_shape = "scalar"
    projection_roles = []
    for role in payload.get("projection_roles") or []:
        role = normalize_text(role)
        if role in PROJECTION_ROLES and role not in projection_roles:
            projection_roles.append(role)
    if operator in {"COUNT", "COUNT_DISTINCT"}:
        projection_roles = [group_role if group_role != "unknown" else "event"]

    mapping_contract = bool(
        content_tokens(question).intersection(MAPPING_CONTRACT_TOKENS)
    )
    paired_roles = [
        role
        for role in (group_role, answer_role)
        if role in GROUP_ROLES and role not in {"event", "identity", "unknown"}
    ]
    if (
        operator in {"COLLECT", "INTERSECTION", "UNION"}
        and result_shape != "tuple_set"
        and mapping_contract
        and len(set(paired_roles)) >= 2
    ):
        result_shape = "tuple_set"
        projection_roles = list(dict.fromkeys(paired_roles))
        warnings.append("normalized_mapping_collection_to_tuple_set")

    if result_shape == "tuple_set":
        role_candidates = list(projection_roles)
        role_candidates.extend(
            role
            for role in (group_role, answer_role)
            if role in PROJECTION_ROLES and role not in {"tuple", "unknown"}
        )
        projection_roles = list(dict.fromkeys(role_candidates))
        if "location" in projection_roles and len(projection_roles) >= 2:
            projection_roles = [
                role for role in projection_roles if role != "location"
            ] + ["location"]
        projection_roles = projection_roles[:2]
        if len(projection_roles) == 1 and projection_roles[0] != "location":
            projection_roles.append("location")
    if not projection_roles:
        if result_shape == "tuple_set":
            primary = group_role if group_role not in {"unknown", "location"} else "object"
            projection_roles = [primary, "location"]
        elif operator in {"COUNT", "COUNT_DISTINCT"}:
            projection_roles = [group_role if group_role != "unknown" else "event"]
        elif answer_role not in {"tuple", "unknown"}:
            projection_roles = [answer_role]
        elif group_role != "unknown":
            projection_roles = [group_role]
    if (
        operator in MULTI_OPERAND_OPERATORS
        and operator != "ELAPSED_TIME"
        and len(operands) < 2
    ):
        if operator == "INTERSECTION" and result_shape in {"tuple_set", "value_set"}:
            operator = "COLLECT"
            warnings.append("corrected_single_set_intersection_to_collect")
        else:
            return _invalid_plan("multi_operand_operator_missing_operands", raw_output)
    if result_shape == "tuple_set" and len(projection_roles) < 2:
        return _invalid_plan("tuple_set_missing_projection_roles", raw_output)
    if confidence != "high":
        warnings.append(f"planner_confidence:{confidence}")
    return SemanticQueryPlan(
        operator=operator,
        answer_type=normalize_text(payload.get("answer_type") or "unknown"),
        target_unit=(str(payload.get("target_unit")).strip().lower()
                     if payload.get("target_unit") is not None else None),
        actor=inferred_actor,
        operands=tuple(operands),
        confidence=confidence,
        status="valid",
        raw_output=raw_output,
        answer_role=answer_role,
        group_role=group_role,
        requested_status=requested_status,
        result_shape=result_shape,
        projection_roles=tuple(projection_roles),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def plan_query_with_local_slm(
    context_layer: Any,
    question: str,
    *,
    include_result_contract: bool = True,
) -> SemanticQueryPlan:
    prompt = semantic_plan_prompt(question, include_result_contract)
    kwargs = {
        "max_tokens": 384,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.05,
        "stop": ["<|end|>", "<|user|>", "\nExplanation:"],
        "echo": False,
    }
    grammar = semantic_plan_grammar(include_result_contract)
    if grammar is not None:
        kwargs["grammar"] = grammar
    try:
        output = context_layer.llm(prompt, **kwargs)
        raw_output = output["choices"][0]["text"].strip()
    except Exception as exc:  # Preserve the legacy path on any planner failure.
        return _invalid_plan(f"planner_call_failed:{type(exc).__name__}")
    return parse_semantic_plan(_extract_json_object(raw_output), question, raw_output)


def _candidate_adjudication_prompt(
    question: str,
    plan: SemanticQueryPlan,
    candidates: list[dict[str, Any]],
    question_timestamp: str | None = None,
    effective_operation: str | None = None,
) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            json.dumps(
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "speaker": candidate.get("source_speaker")
                    or candidate.get("source_role"),
                    "source_timestamp": candidate.get("source_timestamp"),
                    "event_status_hint": candidate.get("event_status_hint")
                    or derive_event_status_hint(
                        candidate.get("clause") or candidate.get("source_quote")
                    ),
                    "value": candidate.get("raw_value")
                    or candidate.get("entity"),
                    "clause": candidate.get("clause"),
                    "source_quote": candidate.get("source_quote"),
                },
                ensure_ascii=True,
            )
        )
    compact_plan = {
        "operator": plan.operator,
        "answer_type": plan.answer_type,
        "target_unit": plan.target_unit,
        "actor": plan.actor,
        "operands": [operand.as_dict() for operand in plan.operands],
    }
    return (
        "<|user|>\nYou adjudicate grounded evidence candidates for one memory "
        "operation. Return JSON only and one decision for every candidate ID. "
        "Do not calculate the final answer. For SUM, COUNT, and COUNT_DISTINCT, "
        "include each individual eligible operand even though that one candidate "
        "cannot answer the aggregate question by itself; completeness is checked "
        "later. Judge actor_match, event_match, time_match, scope_match, and "
        "value_grounded independently. Use not_applicable only when the question "
        "does not constrain that field. Set include=true only when every applicable "
        "field is yes. Planned future events do not satisfy completed-event questions. "
        "You may use stable common taxonomy or geography only to decide whether "
        "a named item or place belongs to the requested scope; the event and "
        "numeric value must still be stated in the candidate. Use the question "
        "and source timestamps to resolve relative time scope. event_status_hint "
        "is a high-precision cue from the source wording; do not change planned "
        "or negated hints to completed. For a requested containing region or "
        "category, a named member that is commonly and stably inside it has "
        "scope_match=yes and scope_basis=stable_taxonomy. Never use taxonomy to "
        "invent an event, value, actor, or date. "
        "canonical_identity must be the exact item or event phrase copied from "
        "source_quote. item_quantity is the number of physical requested items "
        "represented by that phrase; use 1 unless an explicit pair, count, couple, "
        "or dozen states otherwise. Repeated mentions should use the same identity.\n"
        f"Question: {question}\nQuestion timestamp: {question_timestamp or 'unknown'}\n"
        f"Operator: {(effective_operation or plan.operator).upper()}\n"
        f"Plan: {json.dumps(compact_plan, ensure_ascii=True)}\nCandidates:\n"
        + "\n".join(lines)
        + "\n<|end|>\n<|assistant|>\n"
    )


def _explicit_quantity_supported(quote: str, quantity: float) -> bool:
    if quantity == 1:
        return True
    normalized = normalize_text(quote)
    if quantity == 2 and re.search(
        r"\b(?:(?:a|one)\s+)?pair\b|\b(?:a couple|two|2)\b",
        normalized,
    ):
        return True
    if quantity == 12 and re.search(r"\b(?:a dozen|one dozen|12)\b", normalized):
        return True
    return bool(re.search(rf"\b{re.escape(str(int(quantity)))}\b", normalized))


def derive_event_status_hint(text: str) -> str | None:
    """Return only high-precision event-state cues present in the evidence."""
    normalized = normalize_text(text)
    if not normalized:
        return None
    if re.search(
        r"\b(?:did not|didn t|never|cancel(?:ed|led)|not completed|"
        r"not attended|not visited|not booked)\b",
        normalized,
    ):
        return "negated"
    if re.search(
        r"\b(?:plan(?:s|ned|ning)?\s+to|planning\s+(?:a|an|the)|"
        r"intend(?:s|ed|ing)?\s+to|scheduled\s+(?:for|to)|"
        r"going\s+to|will\s+(?:go|visit|attend|take|spend|stay|travel))\b",
        normalized,
    ):
        return "planned"
    if re.search(
        r"\b(?:just\s+got\s+back|returned\s+from|came\s+back\s+from|"
        r"completed|finished|attended|visited|went\s+to|spent|stayed|"
        r"took\s+(?:a|an|the|my)|bought|picked\s+up)\b",
        normalized,
    ):
        return "completed"
    return None


def _grounded_identity_supported(identity: str, candidate: dict[str, Any]) -> bool:
    """Require every identity token to occur in order in the raw evidence.

    Small local models sometimes omit a non-essential modifier when copying an
    item phrase (for example, ``3-day camping trip`` from ``3-day solo camping
    trip``). Exact containment remains the fast path. The bounded ordered-token
    fallback accepts that compression but still rejects invented or reordered
    words.
    """
    identity_tokens = normalize_phrase(identity).split()
    if not identity_tokens:
        return False
    evidence = " ".join(
        str(candidate.get(field) or "")
        for field in ("clause", "source_quote")
    )
    evidence_tokens = normalize_phrase(evidence).split()
    if not evidence_tokens:
        return False
    identity_text = " ".join(identity_tokens)
    evidence_text = " ".join(evidence_tokens)
    if identity_text in evidence_text:
        return True

    matched_positions = []
    cursor = 0
    for token in identity_tokens:
        try:
            position = evidence_tokens.index(token, cursor)
        except ValueError:
            return False
        matched_positions.append(position)
        cursor = position + 1
    inserted_token_count = (
        matched_positions[-1] - matched_positions[0] + 1 - len(identity_tokens)
    )
    return inserted_token_count <= 3


def adjudicate_operation_candidates(
    context_layer: Any,
    question: str,
    plan: SemanticQueryPlan,
    candidates: Iterable[dict[str, Any]],
    *,
    batch_size: int = 1,
    question_timestamp: str | None = None,
    effective_operation: str | None = None,
) -> dict[str, Any]:
    """Review every candidate with constrained output and strict validation."""
    candidates = [
        candidate
        for candidate in candidates or []
        if candidate.get("candidate_id") and candidate.get("source_quote")
    ]
    result = {
        "status": "not_applicable",
        "decisions": [],
        "reviewed_candidate_ids": [],
        "missing_candidate_ids": [],
        "raw_outputs": [],
        "batch_count": 0,
    }
    if not plan.valid or not candidates:
        result["status"] = "ABSENT" if not candidates else "EXTRACTION_FAILED"
        return result

    grammar = candidate_adjudication_grammar()
    decisions = []
    expected_ids = {str(candidate["candidate_id"]).upper() for candidate in candidates}
    candidates_by_id = {
        str(candidate["candidate_id"]).upper(): candidate for candidate in candidates
    }
    for start in range(0, len(candidates), max(1, batch_size)):
        batch = candidates[start : start + max(1, batch_size)]
        prompt = _candidate_adjudication_prompt(
            question,
            plan,
            batch,
            question_timestamp=question_timestamp,
            effective_operation=effective_operation,
        )
        kwargs = {
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.05,
            "stop": ["<|end|>", "<|user|>", "\nExplanation:"],
            "echo": False,
        }
        if grammar is not None:
            kwargs["grammar"] = grammar
        try:
            output = context_layer.llm(prompt, **kwargs)
            raw_output = output["choices"][0]["text"].strip()
        except Exception as exc:
            result.update(
                {
                    "status": "EXTRACTION_FAILED",
                    "error": f"adjudicator_call_failed:{type(exc).__name__}",
                }
            )
            return result
        result["raw_outputs"].append(raw_output)
        result["batch_count"] += 1
        payload = _extract_json_object(raw_output)
        batch_decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(batch_decisions, list):
            result["status"] = "EXTRACTION_FAILED"
            result["error"] = "adjudicator_output_invalid"
            return result
        batch_ids = {
            str(candidate["candidate_id"]).upper() for candidate in batch
        }
        decisions.extend(
            decision
            for decision in batch_decisions
            if isinstance(decision, dict)
            and str(decision.get("candidate_id") or "").upper() in batch_ids
        )

    validated = []
    reviewed = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        candidate_id = str(decision.get("candidate_id") or "").upper()
        if candidate_id not in expected_ids or candidate_id in reviewed:
            continue
        reviewed.add(candidate_id)
        candidate = candidates_by_id[candidate_id]
        declared_include = decision.get("include") is True
        identity = str(decision.get("canonical_identity") or "").strip()
        quantity = decision.get("item_quantity")
        status = normalize_text(decision.get("status"))
        match_fields = {
            field: normalize_text(decision.get(field))
            for field in (
                "actor_match",
                "event_match",
                "time_match",
                "scope_match",
                "value_grounded",
            )
        }
        typed_decision = all(
            value in {"yes", "no", "unknown", "not_applicable"}
            for value in match_fields.values()
        )
        if typed_decision:
            include = all(
                value in {"yes", "not_applicable"}
                for value in match_fields.values()
            )
        else:
            # Compatibility for synthetic/unit callers created before typed
            # adjudication. The constrained local-model schema always supplies
            # the five match fields.
            include = declared_include
        status_hint = candidate.get("event_status_hint") or derive_event_status_hint(
            candidate.get("clause") or candidate.get("source_quote")
        )
        if status_hint in {"planned", "negated"}:
            status = status_hint
        elif status == "unknown" and status_hint == "completed":
            status = status_hint
        if status not in {"completed", "pending", "planned", "negated", "unknown"}:
            include = False
        if include and not _grounded_identity_supported(identity, candidate):
            include = False
        if include:
            quantity_required = plan.operator in {"COUNT", "COUNT_DISTINCT"}
            if not quantity_required:
                quantity = 1
            try:
                quantity = float(quantity if quantity is not None else 1)
            except (TypeError, ValueError):
                include = False
                quantity = None
            if include and (
                quantity <= 0
                or quantity > 1000
                or not _explicit_quantity_supported(candidate["source_quote"], quantity)
            ):
                include = False
        validated.append(
            {
                "candidate_id": candidate_id,
                "include": include,
                "declared_include": declared_include,
                **match_fields,
                "scope_basis": normalize_text(decision.get("scope_basis")),
                "canonical_identity": identity if include else None,
                "item_quantity": quantity if include else None,
                "status": status,
                "source_turn_id": candidate.get("source_turn_id"),
                "source_session_id": candidate.get("source_session_id"),
                "source_quote": candidate.get("source_quote"),
                "candidate": candidate,
            }
        )

    missing = sorted(expected_ids.difference(reviewed))
    result.update(
        {
            "status": "COMPLETE" if not missing else "EXTRACTION_FAILED",
            "decisions": validated,
            "reviewed_candidate_ids": sorted(reviewed),
            "missing_candidate_ids": missing,
        }
    )
    return result


def _format_number(value: float) -> str:
    return str(int(value)) if abs(value - round(value)) < 1e-9 else f"{value:.4f}".rstrip("0").rstrip(".")


def execute_adjudicated_operation_proof(
    plan: SemanticQueryPlan,
    operation: str,
    adjudication: dict[str, Any],
    *,
    minimum_sessions: int = 1,
    target_unit: str | None = None,
) -> OperationProof:
    if adjudication.get("status") != "COMPLETE":
        return OperationProof(
            status="EXTRACTION_FAILED",
            operator=plan.operator,
            answer=None,
            reason=adjudication.get("error") or "candidate_review_incomplete",
        )
    selected = [
        decision
        for decision in adjudication.get("decisions") or []
        if decision.get("include")
        and decision.get("status") not in {"planned", "negated", "unknown"}
    ]
    if not selected:
        return OperationProof(
            status=(
                "PARTIAL"
                if adjudication.get("decisions")
                else "ABSENT"
            ),
            operator=plan.operator,
            answer=None,
            reason=(
                "candidates_present_but_none_eligible"
                if adjudication.get("decisions")
                else "no_evidence_candidates"
            ),
        )

    deduplicated = []
    seen = set()
    for decision in selected:
        identity = normalize_phrase(decision.get("canonical_identity"))
        status = decision.get("status")
        candidate = decision.get("candidate") or {}
        if operation == "count_distinct" or status == "pending":
            fingerprint = (identity, status)
        else:
            fingerprint = (
                identity,
                status,
                candidate.get("source_turn_id"),
                normalize_phrase(candidate.get("clause") or candidate.get("source_quote")),
            )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduplicated.append(decision)

    covered_sessions = {
        decision.get("source_session_id")
        for decision in deduplicated
        if decision.get("source_session_id") is not None
    }
    if len(covered_sessions) < max(1, int(minimum_sessions or 1)):
        return OperationProof(
            status="PARTIAL",
            operator=operation.upper(),
            answer=None,
            reason="minimum_session_coverage_not_met",
            calculation={"covered_session_count": len(covered_sessions)},
        )

    if operation in {"count", "count_distinct"}:
        if operation == "count_distinct":
            value = float(len(deduplicated))
        else:
            value = sum(float(item.get("item_quantity") or 1) for item in deduplicated)
        answer = _format_number(value)
    elif operation == "sum":
        values = []
        units = set()
        for decision in deduplicated:
            candidate = decision.get("candidate") or {}
            value = candidate.get("normalized_value")
            if value is None:
                return OperationProof(
                    status="EXTRACTION_FAILED",
                    operator=plan.operator,
                    answer=None,
                    reason="selected_sum_candidate_lacks_normalized_value",
                )
            values.append(float(value))
            if candidate.get("target_unit"):
                units.add(str(candidate["target_unit"]).rstrip("s"))
        if len(units) > 1:
            return OperationProof(
                status="CONFLICTING",
                operator=plan.operator,
                answer=None,
                reason="incompatible_sum_units",
            )
        value = sum(values)
        unit = (target_unit or (next(iter(units)) if units else None) or "").rstrip("s")
        if unit:
            suffix = unit if abs(value - 1) < 1e-9 else unit + "s"
            answer = f"{_format_number(value)} {suffix}"
        else:
            answer = _format_number(value)
    else:
        return OperationProof(
            status="EXTRACTION_FAILED",
            operator=plan.operator,
            answer=None,
            reason="adjudicated_operator_not_supported",
        )

    return OperationProof(
        status="COMPLETE",
        operator=operation.upper(),
        answer=answer,
        calculation={
            "operator": operation,
            "selected_candidate_ids": [
                decision.get("candidate_id") for decision in deduplicated
            ],
            "covered_session_count": len(covered_sessions),
            "provenance": [
                {
                    "source_turn_id": decision.get("source_turn_id"),
                    "source_session_id": decision.get("source_session_id"),
                    "source_quote": decision.get("source_quote"),
                    "canonical_identity": decision.get("canonical_identity"),
                }
                for decision in deduplicated
            ],
        },
        executable=True,
    )


def _evidence_clauses(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    clauses = [
        part.strip(" ,;:-")
        for part in re.split(r"(?<=[.!?])\s+|\s*;\s*", text)
        if part.strip(" ,;:-")
    ]
    return clauses or [text]


def build_evidence_ledger(
    rows: Iterable[dict[str, Any]],
    temporal_resolver: Callable[[str, str | None], dict[str, Any] | None] | None = None,
    event_status_resolver: Callable[[str], str] | None = None,
) -> list[EvidenceAtom]:
    atoms = []
    seen = set()
    for row in rows or []:
        quote = str(row.get("source_quote") or row.get("text") or "").strip()
        if not quote:
            continue
        turn_ids = row.get("source_turn_ids") or []
        turn_id = row.get("source_turn_id") or (turn_ids[0] if turn_ids else None)
        timestamp = row.get("source_timestamp") or row.get("timestamp")
        for clause_index, clause in enumerate(_evidence_clauses(quote)):
            identity = (
                row.get("source_session_id"),
                turn_id,
                clause_index,
                normalize_text(clause),
            )
            if identity in seen:
                continue
            seen.add(identity)
            temporal_expression = (
                temporal_resolver(clause, timestamp) if temporal_resolver else None
            )
            atoms.append(
                EvidenceAtom(
                    atom_id=f"E{len(atoms) + 1}",
                    source_turn_id=str(turn_id) if turn_id is not None else None,
                    source_session_id=(str(row.get("source_session_id"))
                                       if row.get("source_session_id") is not None else None),
                    source_role=str(row.get("source_role") or row.get("role") or "") or None,
                    source_speaker=str(row.get("source_speaker") or row.get("speaker") or "") or None,
                    source_timestamp=str(timestamp) if timestamp is not None else None,
                    source_quote=quote,
                    clause=clause,
                    clause_index=clause_index,
                    tokens=tuple(sorted(content_tokens(clause))),
                    temporal_expression=temporal_expression,
                    event_status=(event_status_resolver(clause)
                                  if event_status_resolver else None),
                )
            )
    return atoms


FRAME_SUPPORTED_OPERATIONS = {
    "average", "argmax", "argmin", "collect", "collect_distinct", "count",
    "count_distinct", "difference", "earliest", "latest", "sum",
    "temporal_earliest", "temporal_latest",
}


def normalize_measurement_unit(value: Any) -> str | None:
    unit = normalize_phrase(value)
    if not unit:
        return None
    aliases = {
        "$": "usd",
        "currency": "usd",
        "dollar": "usd",
        "dollars": "usd",
        "money": "usd",
        "usd": "usd",
        "mins": "minute",
        "minutes": "minute",
        "minute": "minute",
        "hours": "hour",
        "hour": "hour",
        "days": "day",
        "day": "day",
        "weeks": "week",
        "week": "week",
        "months": "month",
        "month": "month",
        "years": "year",
        "year": "year",
        "items": "item",
        "item": "item",
    }
    return aliases.get(unit, unit[:-1] if unit.endswith("s") else unit)


def _frame_extraction_prompt(
    question: str,
    plan: SemanticQueryPlan,
    atoms: list[EvidenceAtom],
) -> str:
    evidence = [
        {
            "evidence_id": atom.atom_id,
            "speaker": atom.source_speaker or atom.source_role,
            "role": atom.source_role,
            "timestamp": atom.source_timestamp,
            "text": atom.clause,
        }
        for atom in atoms
    ]
    evidence_text = " ".join(atom.clause for atom in atoms)
    compact_operands = []
    for operand in plan.operands:
        payload = operand.as_dict()
        payload["hard_entities"] = [
            entity
            for entity in operand.hard_entities
            if _grounded_identity_supported(
                entity,
                {"clause": evidence_text, "source_quote": evidence_text},
            )
        ]
        compact_operands.append(payload)
    compact_plan = {
        "operator": plan.operator,
        "answer_role": plan.answer_role,
        "group_role": plan.group_role,
        "result_shape": plan.result_shape,
        "projection_roles": list(plan.projection_roles),
        "target_unit": plan.target_unit,
        "actor": plan.actor,
        "requested_status": plan.requested_status,
        "operands": compact_operands,
    }
    return (
        "<|user|>\nCompile each evidence item into one semantic frame. Return "
        "JSON only and exactly one frame for every evidence_id. This is semantic "
        "role labeling, not question answering. Mark relevant=true only when the "
        "evidence states a fact needed by the plan. actor_reference is "
        "source_speaker only when that speaker performs or owns the stated event; "
        "otherwise use other or unknown. Copy predicate, subject, object, "
        "instrument, theme, location, label, event_identity, and event_context "
        "as exact text spans from that evidence item. event_identity is the "
        "concrete event or item that would be counted or listed, never a generic "
        "word such as activity, purchase, event, or context. event_context is the "
        "qualifying topic, project, trip, or scope attached to that identity. Use "
        "null when a role is absent. An instrument is something "
        "used to perform an activity; a theme or object is what the activity is "
        "about or acts on; a label identifies the item associated with a measured "
        "value. Extract only a stated numeric value and its stated unit. quantity "
        "is the explicitly stated number of represented items, otherwise 1. "
        "A hard entity is a constraint only when it appears in the supplied "
        "evidence; an absent result word must not hide evidence that matches the "
        "remaining actor, event, scope, or operand attributes. Status is completed "
        "only for an accomplished event, planned for an "
        "intended future event, negated when denied or cancelled, and unknown "
        "otherwise. Do not use outside knowledge and do not calculate.\n"
        f"Question: {question}\nPlan: {json.dumps(compact_plan, ensure_ascii=True)}\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=True)}\n"
        "<|end|>\n<|assistant|>\n"
    )


def _has_grounded_number(text: str, value: float) -> bool:
    for token in re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?", str(text or "")):
        try:
            if abs(float(token) - float(value)) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def _unit_supported(unit: str | None, atom: EvidenceAtom) -> bool:
    normalized = normalize_measurement_unit(unit)
    if normalized is None:
        return True
    text = normalize_text(atom.clause)
    if normalized == "usd":
        return "$" in atom.clause or any(
            token in content_tokens(text) for token in {"dollar", "usd"}
        )
    return normalized in content_tokens(text)


def _grounded_frame_span(value: Any, atom: EvidenceAtom) -> str | None:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if not value:
        return None
    candidate = {"clause": atom.clause, "source_quote": atom.source_quote}
    return value if _grounded_identity_supported(value, candidate) else None


def _predicate_direct_object(
    atom: EvidenceAtom,
    predicate: str | None,
) -> str | None:
    predicate_tokens = normalize_phrase(predicate).split()
    tokens = _word_tokens(atom.clause)
    if not predicate_tokens or not tokens:
        return None
    lowered = [token["lower"] for token in tokens]
    predicate_start = None
    for index in range(len(tokens) - len(predicate_tokens) + 1):
        if lowered[index : index + len(predicate_tokens)] == predicate_tokens:
            predicate_start = index
            break
    if predicate_start is None:
        return None
    cursor = predicate_start + len(predicate_tokens)
    determiners = {"a", "an", "my", "our", "the", "this", "that"}
    while cursor < len(tokens) and tokens[cursor]["lower"] in determiners:
        cursor += 1
    boundaries = {
        "and", "at", "because", "before", "but", "by", "for", "from",
        "in", "into", "of", "on", "or", "since", "than", "to", "when",
        "while", "with",
    }
    selected = []
    for token in tokens[cursor : cursor + 8]:
        if token["lower"] in boundaries:
            break
        selected.append(token)
    if not selected:
        return None
    return atom.clause[selected[0]["start"] : selected[-1]["end"]]


def _source_actor_reference(
    atom: EvidenceAtom,
    predicate: str | None,
    *,
    strict_subject: bool = False,
) -> str | None:
    tokens = _word_tokens(atom.clause)
    if not tokens:
        return None
    predicate_tokens = normalize_phrase(predicate).split()
    lowered = [token["lower"] for token in tokens]
    predicate_index = len(tokens)
    if predicate_tokens:
        for index in range(len(tokens) - len(predicate_tokens) + 1):
            if lowered[index : index + len(predicate_tokens)] == predicate_tokens:
                predicate_index = index
                break
    subject_tokens = lowered[:predicate_index]
    first_person = {"i", "me", "mine", "my", "our", "ours", "we"}
    if strict_subject:
        # Without a grounded predicate there is no subject boundary. A later
        # possessor or recipient cannot establish who performed the event.
        if not predicate_tokens or predicate_index == len(tokens):
            subject_tokens = lowered[:1]
        first_person = {"i", "we"}
    third_person = {"he", "her", "hers", "him", "his", "she", "their", "they"}
    if set(subject_tokens).intersection(first_person):
        return "source_speaker"
    if set(subject_tokens).intersection(third_person):
        return "other"
    return None


def _frame_candidate_atoms(
    plan: SemanticQueryPlan,
    question: str,
    ledger: Iterable[EvidenceAtom],
    limit: int,
) -> list[EvidenceAtom]:
    query_tokens = content_tokens(question)
    for operand in plan.operands:
        query_tokens.update(content_tokens(operand.text))
    numeric_operation = plan.operator in {
        "AVERAGE", "ARGMAX", "ARGMIN", "DIFFERENCE", "SUM",
    }
    temporal_extreme = plan.operator in {"EARLIEST", "LATEST"}
    ranked = []
    for index, atom in enumerate(ledger or []):
        if plan.actor and not _actor_matches(plan.actor, atom):
            continue
        overlap = len(query_tokens.intersection(set(atom.tokens)))
        has_number = bool(re.search(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", atom.clause))
        source_temporal = _source_timestamp_temporal(atom) if temporal_extreme else None
        if overlap == 0 and not (
            (numeric_operation and has_number)
            or (temporal_extreme and source_temporal is not None)
        ):
            continue
        date_ordinal = 0
        if source_temporal is not None:
            try:
                date_ordinal = datetime.date.fromisoformat(
                    source_temporal["value"]
                ).toordinal()
            except (KeyError, TypeError, ValueError):
                date_ordinal = 0
            if plan.operator == "EARLIEST":
                date_ordinal = -date_ordinal
        secondary = date_ordinal if temporal_extreme else int(has_number)
        ranked.append((overlap, secondary, int(has_number), -index, atom))
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return [item[4] for item in ranked[: max(1, int(limit or 1))]]


def compile_evidence_frames(
    context_layer: Any,
    question: str,
    plan: SemanticQueryPlan,
    ledger: Iterable[EvidenceAtom],
    *,
    batch_size: int = 1,
    max_atoms: int = 24,
) -> tuple[list[EvidenceFrame], dict[str, Any]]:
    atoms = _frame_candidate_atoms(plan, question, ledger, max_atoms)
    diagnostics = {
        "applicable": effective_semantic_operator(plan).lower()
        in FRAME_SUPPORTED_OPERATIONS,
        "candidate_atom_count": len(atoms),
        "reviewed_atom_count": 0,
        "batch_count": 0,
        "missing_evidence_ids": [],
        "invalid_span_count": 0,
        "invalid_value_count": 0,
        "status": "ABSENT" if not atoms else "PENDING",
    }
    if not diagnostics["applicable"] or not atoms:
        return [], diagnostics

    grammar = evidence_frame_grammar()
    frames = []
    atoms_by_id = {atom.atom_id.upper(): atom for atom in atoms}
    reviewed = set()
    for start in range(0, len(atoms), max(1, batch_size)):
        batch = atoms[start : start + max(1, batch_size)]
        kwargs = {
            "max_tokens": 384,
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.05,
            "stop": ["<|end|>", "<|user|>", "\nExplanation:"],
            "echo": False,
        }
        if grammar is not None:
            kwargs["grammar"] = grammar
        try:
            output = context_layer.llm(
                _frame_extraction_prompt(question, plan, batch),
                **kwargs,
            )
            raw_output = output["choices"][0]["text"].strip()
        except Exception as exc:
            diagnostics.update(
                {
                    "status": "EXTRACTION_FAILED",
                    "error": f"frame_extractor_call_failed:{type(exc).__name__}",
                }
            )
            return frames, diagnostics
        diagnostics["batch_count"] += 1
        payload = _extract_json_object(raw_output)
        raw_frames = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(raw_frames, list):
            diagnostics.update(
                {"status": "EXTRACTION_FAILED", "error": "frame_output_invalid"}
            )
            return frames, diagnostics
        batch_ids = {atom.atom_id.upper() for atom in batch}
        for raw_frame in raw_frames:
            if not isinstance(raw_frame, dict):
                continue
            evidence_id = str(raw_frame.get("evidence_id") or "").upper()
            if evidence_id not in batch_ids or evidence_id in reviewed:
                continue
            reviewed.add(evidence_id)
            atom = atoms_by_id[evidence_id]
            spans = {}
            for field in (
                "predicate", "subject", "object", "instrument", "theme",
                "location", "label", "event_identity", "event_context",
            ):
                raw_span = raw_frame.get(field)
                spans[field] = _grounded_frame_span(raw_span, atom)
                if raw_span and spans[field] is None:
                    diagnostics["invalid_span_count"] += 1
            value = raw_frame.get("value")
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is None:
                grounded_numbers = []
                for field in ("object", "label", "subject", "theme"):
                    for token in re.findall(
                        r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?",
                        str(spans.get(field) or ""),
                    ):
                        try:
                            grounded_numbers.append(float(token))
                        except ValueError:
                            continue
                if len(set(grounded_numbers)) == 1:
                    value = grounded_numbers[0]
            if value is not None and not _has_grounded_number(atom.clause, value):
                diagnostics["invalid_value_count"] += 1
                value = None
            unit = normalize_measurement_unit(raw_frame.get("unit"))
            if unit is None:
                if "$" in atom.clause:
                    unit = "usd"
                else:
                    for candidate_unit in (
                        "minute", "hour", "day", "week", "month", "year",
                    ):
                        if candidate_unit in content_tokens(atom.clause):
                            unit = candidate_unit
                            break
            if unit and not _unit_supported(unit, atom):
                unit = None
            try:
                quantity = float(raw_frame.get("quantity", 1))
            except (TypeError, ValueError):
                quantity = 1.0
            if (
                quantity <= 0
                or quantity > 1000
                or not _explicit_quantity_supported(atom.clause, quantity)
            ):
                quantity = 1.0
            status = normalize_text(raw_frame.get("status"))
            status_hint = atom.event_status or derive_event_status_hint(atom.clause)
            if status_hint in {"negated", "negated_or_cancelled"}:
                status = "negated"
            elif status_hint in {"planned", "pending"}:
                status = "planned"
            elif status_hint == "completed":
                status = "completed"
            if status not in {"completed", "planned", "negated", "unknown"}:
                status = "unknown"
            actor_reference = normalize_text(raw_frame.get("actor_reference"))
            if actor_reference not in {"source_speaker", "other", "unknown"}:
                actor_reference = "unknown"
            structural_actor = _source_actor_reference(
                atom,
                spans["predicate"],
                strict_subject=plan.operator in {"COUNT", "COUNT_DISTINCT"},
            )
            if structural_actor is not None:
                actor_reference = structural_actor
            elif (
                actor_reference == "source_speaker"
                and normalize_text(atom.source_role) == "user"
                and normalize_text(
                    plan.actor
                    or next(
                        (operand.actor for operand in plan.operands if operand.actor),
                        None,
                    )
                    or "user"
                ) in {"i", "me", "mine", "my", "speaker user", "user"}
            ):
                # A user-role turn is not automatically evidence that the user
                # owns the event.  Without a first-person grammatical subject,
                # keep the actor unresolved so strict aggregate/tuple proofs
                # cannot absorb third-party or background statements.
                actor_reference = "unknown"
            predicate_lemma = normalize_text(spans["predicate"])
            if predicate_lemma in {"use", "used", "using"} and not spans["instrument"]:
                spans["instrument"] = _predicate_direct_object(
                    atom,
                    spans["predicate"],
                )
            frames.append(
                EvidenceFrame(
                    frame_id=f"F{len(frames) + 1}",
                    evidence_id=atom.atom_id,
                    relevant=raw_frame.get("relevant") is True,
                    actor_reference=actor_reference,
                    status=status,
                    value=value,
                    unit=unit,
                    quantity=quantity,
                    source_turn_id=atom.source_turn_id,
                    source_session_id=atom.source_session_id,
                    source_role=atom.source_role,
                    source_speaker=atom.source_speaker,
                    source_timestamp=atom.source_timestamp,
                    source_quote=atom.source_quote,
                    clause=atom.clause,
                    provenance_identity=atom.provenance_identity,
                    **spans,
                )
            )

    missing = sorted(set(atoms_by_id).difference(reviewed))
    diagnostics["reviewed_atom_count"] = len(reviewed)
    diagnostics["missing_evidence_ids"] = missing
    diagnostics["status"] = "COMPLETE" if not missing else "EXTRACTION_FAILED"
    return frames, diagnostics


def _frame_scope_tokens(frame: EvidenceFrame) -> set[str]:
    return content_tokens(frame.clause).union(
        *(
            content_tokens(getattr(frame, field))
            for field in (
                "predicate", "object", "instrument", "theme", "location", "label",
                "event_identity", "event_context",
            )
        )
    )


TEMPORAL_EXTREME_SCOPE_FILLER = {
    "activity", "chronological", "completed", "confirm", "conversation",
    "earliest", "end", "event", "first", "history", "last", "latest",
    "new", "newest", "oldest", "point", "record", "separate", "thing",
    "timeline", "update",
}


def _semantic_frame_scope_tokens(
    plan: SemanticQueryPlan,
    question: str,
) -> set[str]:
    """Return only the semantic subject scope that evidence must share.

    Latest/earliest questions may either name a real domain (for example,
    "latest repair") or ask for the global extreme ("latest thing I did").
    The latter has no lexical subject constraint: actor, event status, and
    provenance time are the complete contract. Explicit hard entities always
    remain mandatory, while generic intent words cannot accidentally filter
    every dated event out of a global query.
    """
    if plan.operator in {"COUNT", "COUNT_DISTINCT"}:
        return set(count_scope_contract(plan, question)["topic_tokens"])
    hard_scope = set().union(
        *(
            content_tokens(entity)
            for operand in plan.operands
            for entity in operand.hard_entities
        )
    )
    if hard_scope:
        domain_scope = hard_scope.difference(
            content_tokens(" ".join(SEMANTIC_CONTRACT_TOKENS))
        )
        return domain_scope or hard_scope
    scope = set().union(
        content_tokens(question),
        *(content_tokens(operand.text) for operand in plan.operands),
    )
    if plan.operator in {"EARLIEST", "LATEST"}:
        scope.difference_update(TEMPORAL_EXTREME_SCOPE_FILLER)
    domain_scope = scope.difference(
        content_tokens(" ".join(SEMANTIC_CONTRACT_TOKENS))
    )
    return domain_scope or scope


def count_scope_contract(plan: SemanticQueryPlan, question: str) -> dict[str, Any]:
    """Separate the counted entity type from its question-grounded topic."""
    control = content_tokens(" ".join(
        SEMANTIC_CONTRACT_TOKENS | COUNT_SCOPE_CONTROL_WORDS | GENERIC_ROLE_TOKENS
    ))
    hard_scope = set().union(*(
        content_tokens(entity)
        for operand in plan.operands
        for entity in operand.hard_entities
    ))
    domain = hard_scope.difference(control)
    source = "hard_entities"
    if not domain:
        domain = set().union(*(
            content_tokens(operand.text) for operand in plan.operands
        )).difference(control)
        source = "operands"
    if not domain:
        domain = content_tokens(question).difference(control)
        source = "question"
    return {
        "topic_tokens": sorted(domain), "topic_source": source,
        "counted_role": plan.group_role,
        "discarded_type_tokens": sorted(hard_scope.intersection(control)),
        "policy": "counted_role_is_separate_from_topic",
    }


def expand_count_compiler_sources(
    plan: SemanticQueryPlan,
    question: str,
    selected_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve scoped sources already present in the bounded retrieved pool."""
    rows = list(selected_rows or [])
    diagnostics = {"applicable": plan.operator in {"COUNT", "COUNT_DISTINCT"}}
    if not diagnostics["applicable"]:
        return rows, diagnostics
    scope = count_scope_contract(plan, question)
    topic = set(scope["topic_tokens"])
    actor = plan.actor or next(
        (operand.actor for operand in plan.operands if operand.actor), None
    ) or "user"
    candidates = list(candidate_rows or [])
    admitted = []
    rejected = defaultdict(int)
    for candidate in candidates:
        atoms = build_evidence_ledger([candidate])
        if not any(_actor_matches(actor, atom) for atom in atoms):
            rejected["source_actor"] += 1
            continue
        if not any(_scope_contract_matches(topic, set(atom.tokens)) for atom in atoms):
            rejected["topic"] += 1
            continue
        admitted.append(candidate)
    selected_sessions = {
        str(row["source_session_id"])
        for row in rows if row.get("source_session_id") is not None
    }
    admitted_sessions = {
        str(row["source_session_id"])
        for row in admitted if row.get("source_session_id") is not None
    }
    seen = set()
    expanded = []
    for candidate in rows + admitted:
        turn_ids = candidate.get("source_turn_ids") or []
        turn_id = candidate.get("source_turn_id")
        if turn_id is None and len(turn_ids) == 1:
            turn_id = turn_ids[0]
        identity = (
            str(candidate.get("source_session_id") or ""),
            str(turn_id) if turn_id is not None else tuple(map(str, turn_ids)),
            normalize_text(candidate.get("source_role") or candidate.get("role")),
            normalize_text(candidate.get("source_speaker") or candidate.get("speaker")),
            normalize_text(candidate.get("source_quote") or candidate.get("text")),
        )
        if identity not in seen:
            seen.add(identity)
            expanded.append(candidate)
    represented = {
        atom.source_session_id for atom in build_evidence_ledger(expanded)
        if atom.source_session_id is not None
    }
    diagnostics.update({
        "scope_contract": scope, "candidate_row_count": len(candidates),
        "admitted_candidate_count": len(admitted),
        "selected_row_count": len(rows), "expanded_row_count": len(expanded),
        "admitted_session_ids": sorted(admitted_sessions),
        "restored_session_ids": sorted(admitted_sessions - selected_sessions),
        "unrepresented_candidate_session_ids": sorted(admitted_sessions - represented),
        "retrieved_scope_sources_preserved": admitted_sessions.issubset(represented),
        "retrieval_exhaustive": False, "rejected_counts": dict(rejected),
        "policy": "preserve_scoped_retrieved_sources_before_count_execution",
    })
    return expanded, diagnostics


def _scope_contract_matches(
    scope_tokens: set[str],
    evidence_tokens: set[str],
) -> bool:
    """Require substantive scope coverage, not one generic shared token."""
    if not scope_tokens:
        return True
    overlap = scope_tokens.intersection(evidence_tokens)
    if not overlap:
        return False
    if len(scope_tokens) <= 2:
        return True
    return len(overlap) >= 2 or len(overlap) / len(scope_tokens) >= 0.5


def _is_generic_role_value(
    value: str | None,
    scope_tokens: set[str] | None = None,
) -> bool:
    tokens = content_tokens(value)
    if not tokens:
        return True
    generic = content_tokens(" ".join(GENERIC_ROLE_TOKENS))
    residual = tokens.difference(generic)
    if scope_tokens:
        residual.difference_update(scope_tokens)
    return not residual


def _value_without_location(value: str | None, location: str | None) -> str | None:
    """Remove an already-bound location span from a candidate item span."""
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    location_tokens = [token["lower"] for token in _word_tokens(location or "")]
    value_tokens = _word_tokens(value)
    if not value or not location_tokens or not value_tokens:
        return value or None
    lowered = [token["lower"] for token in value_tokens]
    for index in range(len(lowered) - len(location_tokens) + 1):
        if lowered[index : index + len(location_tokens)] != location_tokens:
            continue
        end = index
        if end > 0 and lowered[end - 1] in {"at", "in", "near", "on"}:
            end -= 1
        kept = value_tokens[:end] + value_tokens[index + len(location_tokens) :]
        if not kept:
            return None
        return " ".join(token["raw"] for token in kept).strip() or None
    return value


def _frame_role_value(
    frame: EvidenceFrame,
    role: str,
    scope_tokens: set[str] | None = None,
) -> str | None:
    role = normalize_text(role)
    if role == "object":
        candidates = (
            frame.object,
            frame.instrument,
            frame.label,
            _value_without_location(frame.subject, frame.location),
            frame.event_identity,
        )
        return next(
            (
                value for value in candidates
                if value and not _is_generic_role_value(value, scope_tokens)
            ),
            None,
        )
    if role in {"instrument", "label", "location", "theme"}:
        return getattr(frame, role)
    if role in {"identity", "unknown"}:
        if frame.event_identity and not _is_generic_role_value(
            frame.event_identity, scope_tokens
        ):
            return frame.event_identity
        values = [
            value
            for value in (
                frame.label, frame.instrument, frame.object, frame.theme,
            )
            if value
        ]
        return values[0] if len(values) == 1 else None
    if role == "event":
        if frame.event_identity and not _is_generic_role_value(
            frame.event_identity, scope_tokens
        ):
            return frame.event_identity
        parts = [
            frame.predicate,
            frame.object or frame.instrument or frame.label or frame.theme,
        ]
        value = " ".join(part for part in parts if part).strip()
        return value if value and not _is_generic_role_value(
            value, scope_tokens
        ) else None
    return None


def _frame_projection_values(
    frame: EvidenceFrame,
    projection_roles: Iterable[str],
    scope_tokens: set[str] | None = None,
) -> list[str] | None:
    values = []
    for role in projection_roles:
        value = _frame_role_value(frame, role, scope_tokens)
        if not value:
            return None
        values.append(value)
    return values


def _format_measurement(value: float, unit: str | None) -> str:
    if unit == "usd":
        return f"${_format_number(value)}"
    if unit:
        suffix = unit if abs(value - 1) < 1e-9 else unit + "s"
        return f"{_format_number(value)} {suffix}"
    return _format_number(value)


def _frame_source_date(frame: EvidenceFrame) -> datetime.date | None:
    """Parse only an explicit calendar date carried by frame provenance."""
    match = re.search(
        r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)",
        str(frame.source_timestamp or ""),
    )
    if not match:
        return None
    try:
        return datetime.date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _candidate_actor_matches_source(
    candidate: dict[str, Any],
    requested_actor: str,
) -> bool:
    role = normalize_text(candidate.get("source_role"))
    speaker = normalize_text(candidate.get("source_speaker"))
    requested = normalize_text(requested_actor)
    if requested in {"i", "me", "my", "speaker user", "user"}:
        if role != "user" and speaker != "user":
            return False
    elif requested in {"assistant", "speaker assistant"}:
        if role != "assistant" and speaker != "assistant":
            return False
    elif requested != speaker:
        return False

    clause = str(
        candidate.get("clause")
        or candidate.get("sentence")
        or candidate.get("source_quote")
        or ""
    )
    tokens = _word_tokens(clause)
    if not tokens:
        return False
    first_person = {"i", "me", "mine", "my", "our", "ours", "we"}
    if any(token["lower"] in first_person for token in tokens[:12]):
        return True
    third_person = {"he", "her", "hers", "him", "his", "she", "their", "they"}
    if any(token["lower"] in third_person for token in tokens[:6]):
        return False
    predicates = {
        part
        for definition in (
            list(COMPLETED_EVENT_PREDICATES) + list(PENDING_EVENT_PREDICATES)
        )
        for part in definition
    } | {"cost", "paid", "recorded", "spent", "used"}
    sentence_starters = {"a", "an", "the", "this", "that", "these", "those"}
    for index, token in enumerate(tokens[:6]):
        if token["lower"] in sentence_starters or not token["raw"][:1].isupper():
            continue
        if index + 1 < len(tokens) and tokens[index + 1]["lower"] in predicates:
            return False
    return True


def _candidate_measurement_label(
    candidate: dict[str, Any],
    value: float,
) -> str | None:
    clause = str(
        candidate.get("clause")
        or candidate.get("sentence")
        or candidate.get("source_quote")
        or ""
    )
    tokens = _word_tokens(clause)
    value_index = None
    for index, token in enumerate(tokens):
        try:
            numeric = float(token["raw"].replace(",", ""))
        except ValueError:
            continue
        if abs(numeric - value) < 1e-9:
            value_index = index
            break
    if value_index is None:
        return None
    linking_verbs = {"are", "cost", "costs", "is", "was", "were"}
    boundary = value_index
    for index in range(value_index - 1, max(-1, value_index - 5), -1):
        if tokens[index]["lower"] in linking_verbs:
            boundary = index
            break
    separators = {"at", "by", "for", "from", "of", "on", "with"}
    start = None
    for index in range(boundary - 1, max(-1, boundary - 8), -1):
        if tokens[index]["lower"] in separators:
            start = index + 1
            break
    if start is None or start >= boundary:
        return None
    selected = tokens[start:boundary]
    if not selected:
        return None
    return clause[selected[0]["start"] : selected[-1]["end"]]


SEMANTIC_OPERAND_MIN_SCORE = 0.30
SEMANTIC_OPERAND_MIN_MARGIN = 0.08
SEMANTIC_OPERAND_CANDIDATE_MARGIN = 0.04


def _semantic_operand_scores(candidate: dict[str, Any]) -> dict[int, float]:
    scores = {}
    for row in candidate.get("semantic_operand_scores") or []:
        try:
            index = int(row.get("operand_index"))
            score = float(row.get("score"))
        except (AttributeError, TypeError, ValueError):
            continue
        scores[index] = score
    return scores


def _bind_grounded_candidates_to_operands(
    candidates: list[dict[str, Any]],
    operand_count: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Bind one grounded candidate to each operand, conservatively.

    Lexical bindings retain priority. Dense scores may fill only uncovered
    operands, and only when both the per-candidate operand margin and the
    per-operand candidate margin are unambiguous.
    """
    diagnostics = {
        "required_operand_count": operand_count,
        "lexical_bindings": {},
        "semantic_bindings": {},
        "unresolved_operand_indexes": [],
        "ambiguous_operand_indexes": [],
    }
    lexical = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        try:
            operand_index = int(candidate.get("operand_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= operand_index < operand_count:
            lexical[operand_index].append(candidate_index)

    selected_by_operand: dict[int, int] = {}
    used_candidates = set()
    for operand_index, candidate_indexes in lexical.items():
        if len(candidate_indexes) != 1:
            diagnostics["ambiguous_operand_indexes"].append(operand_index)
            continue
        candidate_index = candidate_indexes[0]
        selected_by_operand[operand_index] = candidate_index
        used_candidates.add(candidate_index)
        diagnostics["lexical_bindings"][str(operand_index)] = candidate_index

    if diagnostics["ambiguous_operand_indexes"]:
        return None, diagnostics

    unresolved = [
        index for index in range(operand_count) if index not in selected_by_operand
    ]
    semantic_options = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index in used_candidates:
            continue
        scores = _semantic_operand_scores(candidate)
        if not scores:
            continue
        # Compare only the operands that remain available.  Once a strong
        # lexical binding has claimed (for example) the taxi operand, a train
        # candidate should be judged against the unresolved rail operand rather
        # than rejected because both transport embeddings are naturally close.
        ranked = sorted(
            (index for index in scores if index in unresolved),
            key=scores.__getitem__,
            reverse=True,
        )
        if not ranked:
            continue
        best_operand = ranked[0]
        best_score = scores[best_operand]
        runner_up = scores[ranked[1]] if len(ranked) > 1 else -1.0
        if (
            best_operand not in unresolved
            or best_score < SEMANTIC_OPERAND_MIN_SCORE
            or best_score - runner_up < SEMANTIC_OPERAND_MIN_MARGIN
        ):
            continue
        semantic_options[best_operand].append((best_score, candidate_index))

    for operand_index in unresolved:
        options = sorted(semantic_options.get(operand_index, []), reverse=True)
        if not options:
            diagnostics["unresolved_operand_indexes"].append(operand_index)
            continue
        if (
            len(options) > 1
            and options[0][0] - options[1][0]
            < SEMANTIC_OPERAND_CANDIDATE_MARGIN
        ):
            diagnostics["ambiguous_operand_indexes"].append(operand_index)
            continue
        score, candidate_index = options[0]
        if candidate_index in used_candidates:
            diagnostics["ambiguous_operand_indexes"].append(operand_index)
            continue
        selected_by_operand[operand_index] = candidate_index
        used_candidates.add(candidate_index)
        diagnostics["semantic_bindings"][str(operand_index)] = {
            "candidate_index": candidate_index,
            "score": score,
        }

    if (
        diagnostics["unresolved_operand_indexes"]
        or diagnostics["ambiguous_operand_indexes"]
        or len(selected_by_operand) != operand_count
    ):
        return None, diagnostics

    selected = []
    for operand_index in range(operand_count):
        candidate = dict(candidates[selected_by_operand[operand_index]])
        candidate["bound_operand_index"] = operand_index
        candidate["operand_binding_method"] = (
            "lexical"
            if str(operand_index) in diagnostics["lexical_bindings"]
            else "semantic"
        )
        selected.append(candidate)
    return selected, diagnostics


def execute_grounded_candidate_operation_proof(
    plan: SemanticQueryPlan,
    candidates: Iterable[dict[str, Any]],
    *,
    effective_operation: str,
    minimum_sessions: int = 1,
    target_unit: str | None = None,
) -> tuple[OperationProof | None, dict[str, Any]]:
    """Execute numeric operations from adapter-grounded candidate records.

    Candidate extraction has already located and normalized explicit numbers.
    This executor independently rechecks provenance, grounding, scope, actor,
    event state, and units before doing arithmetic. It never asks the language
    model to select values or calculate a result.
    """
    operation = normalize_text(effective_operation)
    supported = {"average", "argmax", "argmin", "difference", "sum"}
    diagnostics = {
        "applicable": operation in supported,
        "candidate_count": 0,
        "eligible_candidate_count": 0,
        "rejected_counts": {},
        "scope_tokens": [],
        "candidate_audit": [],
    }
    if operation not in supported:
        return None, diagnostics
    candidates = list(candidates or [])
    diagnostics["candidate_count"] = len(candidates)
    if not candidates:
        return OperationProof(
            status="ABSENT", operator=plan.operator, answer=None,
            reason="no_grounded_numeric_candidates",
        ), diagnostics

    requested_actor = plan.actor or next(
        (operand.actor for operand in plan.operands if operand.actor),
        None,
    ) or "user"
    requested_status = plan.requested_status
    requires_operand_binding = (
        len(plan.operands) > 1
        and operation in {"argmax", "argmin", "average", "difference", "sum"}
    )
    raw_scope_tokens = set().union(
        *(content_tokens(operand.text) for operand in plan.operands)
    )
    scope_tokens = raw_scope_tokens.difference(
        content_tokens(" ".join(SEMANTIC_CONTRACT_TOKENS))
    ) or raw_scope_tokens
    diagnostics["scope_tokens"] = sorted(scope_tokens)
    rejected = defaultdict(int)
    eligible = []
    seen = set()
    for candidate in candidates:
        audit = {
            "candidate_id": candidate.get("candidate_id"),
            "source_turn_id": candidate.get("source_turn_id"),
            "source_session_id": candidate.get("source_session_id"),
            "decision": "rejected",
            "reason": None,
        }
        source_quote = str(candidate.get("source_quote") or "")
        clause = str(
            candidate.get("clause") or candidate.get("sentence") or source_quote
        )
        if not _candidate_actor_matches_source(candidate, requested_actor):
            rejected["actor"] += 1
            audit["reason"] = "actor"
            diagnostics["candidate_audit"].append(audit)
            continue
        status = normalize_text(
            derive_event_status_hint(clause)
            or candidate.get("event_status_hint")
            or derive_event_status_hint(source_quote)
            or "unknown"
        )
        if status == "negated_or_cancelled":
            status = "negated"
        if requested_status == "completed" and status != "completed":
            rejected["status"] += 1
            audit.update({"reason": "status", "status": status})
            diagnostics["candidate_audit"].append(audit)
            continue
        if requested_status == "planned" and status not in {"planned", "pending"}:
            rejected["status"] += 1
            audit.update({"reason": "status", "status": status})
            diagnostics["candidate_audit"].append(audit)
            continue
        if requested_status in {"any", "unknown"} and status in {
            "planned", "pending", "negated",
        }:
            rejected["status"] += 1
            audit.update({"reason": "status", "status": status})
            diagnostics["candidate_audit"].append(audit)
            continue
        candidate_tokens = content_tokens(clause)
        scope_overlap = sorted(scope_tokens.intersection(candidate_tokens))
        lexical_scope_match = _scope_contract_matches(
            scope_tokens,
            candidate_tokens,
        )
        semantic_scores = _semantic_operand_scores(candidate)
        audit.update(
            {
                "status": status,
                "scope_overlap": scope_overlap,
                "scope_coverage": (
                    len(scope_overlap) / len(scope_tokens) if scope_tokens else 1.0
                ),
                "semantic_operand_scores": semantic_scores,
            }
        )
        semantic_scope_match = bool(
            requires_operand_binding
            and semantic_scores
            and max(semantic_scores.values()) >= SEMANTIC_OPERAND_MIN_SCORE
        )
        if scope_tokens and not lexical_scope_match and not semantic_scope_match:
            rejected["scope"] += 1
            audit["reason"] = "scope"
            diagnostics["candidate_audit"].append(audit)
            continue
        raw_value = candidate.get("normalized_value")
        if raw_value is None:
            raw_value = candidate.get("numeric_value")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            rejected["value"] += 1
            audit["reason"] = "value"
            diagnostics["candidate_audit"].append(audit)
            continue
        if not _has_grounded_number(clause + " " + source_quote, value):
            rejected["value_grounding"] += 1
            audit["reason"] = "value_grounding"
            diagnostics["candidate_audit"].append(audit)
            continue
        unit = normalize_measurement_unit(
            candidate.get("target_unit")
            or candidate.get("unit")
            or candidate.get("currency")
        )
        identity = (
            candidate.get("source_session_id"),
            candidate.get("source_turn_id"),
            value,
            unit,
        )
        if identity in seen:
            audit.update({"reason": "duplicate_claim_value", "value": value, "unit": unit})
            diagnostics["candidate_audit"].append(audit)
            continue
        seen.add(identity)
        label = None
        for field in ("label", "entity", "grounded_identity", "head"):
            proposed = str(candidate.get(field) or "").strip()
            if proposed and _grounded_identity_supported(
                proposed,
                {"clause": clause, "source_quote": source_quote},
            ):
                label = proposed
                break
        if label is None:
            label = _candidate_measurement_label(candidate, value)
        eligible.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "value": value,
                "unit": unit,
                "label": label,
                "source_turn_id": candidate.get("source_turn_id"),
                "source_session_id": candidate.get("source_session_id"),
                "source_quote": source_quote,
                "clause": clause,
                "status": status,
                "operand_index": candidate.get("operand_index"),
                "operand_match_score": candidate.get("operand_match_score"),
                "semantic_operand_scores": candidate.get(
                    "semantic_operand_scores"
                ),
            }
        )
        audit.update(
            {
                "decision": "eligible",
                "reason": None,
                "value": value,
                "unit": unit,
                "label": label,
            }
        )
        diagnostics["candidate_audit"].append(audit)

    diagnostics["eligible_candidate_count"] = len(eligible)
    diagnostics["rejected_counts"] = dict(rejected)
    if not eligible:
        return OperationProof(
            status="PARTIAL", operator=plan.operator, answer=None,
            reason="no_eligible_grounded_numeric_candidates",
        ), diagnostics
    selected = eligible
    if requires_operand_binding:
        selected, binding_diagnostics = _bind_grounded_candidates_to_operands(
            eligible,
            len(plan.operands),
        )
        diagnostics["operand_binding"] = binding_diagnostics
        if selected is None:
            return OperationProof(
                status="PARTIAL", operator=plan.operator, answer=None,
                reason="multi_operand_binding_incomplete_or_ambiguous",
            ), diagnostics
        diagnostics["selected_operand_bindings"] = [
            {
                "candidate_id": candidate.get("candidate_id"),
                "bound_operand_index": candidate.get("bound_operand_index"),
                "method": candidate.get("operand_binding_method"),
                "value": candidate.get("value"),
            }
            for candidate in selected
        ]
    covered_sessions = {
        candidate["source_session_id"]
        for candidate in selected
        if candidate["source_session_id"] is not None
    }
    if len(covered_sessions) < max(1, int(minimum_sessions or 1)):
        return OperationProof(
            status="PARTIAL", operator=plan.operator, answer=None,
            reason="minimum_session_coverage_not_met",
            calculation={"covered_session_count": len(covered_sessions)},
        ), diagnostics

    units = {candidate["unit"] for candidate in selected if candidate["unit"]}
    normalized_target = normalize_measurement_unit(target_unit or plan.target_unit)
    if normalized_target:
        incompatible = {unit for unit in units if unit != normalized_target}
        if incompatible:
            return OperationProof(
                status="CONFLICTING", operator=plan.operator, answer=None,
                reason="incompatible_measurement_units",
            ), diagnostics
        unit = normalized_target
    elif len(units) <= 1:
        unit = next(iter(units), None)
    else:
        return OperationProof(
            status="CONFLICTING", operator=plan.operator, answer=None,
            reason="incompatible_measurement_units",
        ), diagnostics

    values = [candidate["value"] for candidate in selected]
    if operation == "sum":
        result = sum(values)
        answer = _format_measurement(result, unit)
    elif operation == "average":
        result = sum(values) / len(values)
        answer = _format_measurement(result, unit)
    elif operation == "difference":
        if len(values) != 2:
            return OperationProof(
                status="PARTIAL", operator=plan.operator, answer=None,
                reason="difference_requires_two_grounded_values",
            ), diagnostics
        result = abs(values[0] - values[1])
        answer = _format_measurement(result, unit)
    else:
        chooser = max if operation == "argmax" else min
        extreme = chooser(selected, key=lambda candidate: candidate["value"])
        if not extreme["label"]:
            return OperationProof(
                status="PARTIAL", operator=plan.operator, answer=None,
                reason="extreme_value_label_missing",
            ), diagnostics
        result = extreme["value"]
        answer = extreme["label"]

    return OperationProof(
        status="COMPLETE",
        operator=plan.operator,
        answer=answer,
        calculation={
            "operator": operation,
            "executor": "deterministic_grounded_candidate_executor",
            "values": values,
            "result": result,
            "unit": unit,
            "covered_session_count": len(covered_sessions),
            "provenance": selected,
        },
        executable=True,
    ), diagnostics


def _frame_requested_identity(
    frame: EvidenceFrame,
    plan: SemanticQueryPlan,
    scope_tokens: set[str],
) -> str | None:
    value = _frame_role_value(frame, plan.group_role, scope_tokens)
    if value:
        return value
    if plan.group_role in {"event", "identity", "unknown"} and (
        frame.source_session_id is not None and frame.source_turn_id is not None
    ):
        return (
            f"claim:{frame.source_session_id}:{frame.source_turn_id}:"
            f"{normalize_phrase(frame.clause)}"
        )
    return None


def _frame_selection_quality(
    frame: EvidenceFrame,
    plan: SemanticQueryPlan,
    operation: str,
    scope_tokens: set[str],
    projection_roles: tuple[str, ...],
) -> tuple[float, int, int]:
    frame_tokens = _frame_scope_tokens(frame)
    overlap = len(scope_tokens.intersection(frame_tokens))
    score = float(overlap * 4)
    if frame.actor_reference == "source_speaker":
        score += 6
    if frame.relevant:
        score += 4
    if operation == "collect_distinct":
        projected = _frame_projection_values(frame, projection_roles, scope_tokens)
        score += 8 if projected else 0
    elif operation == "count_distinct":
        score += 6 if _frame_requested_identity(frame, plan, scope_tokens) else 0
    elif frame.value is not None:
        score += 6
    populated_roles = sum(
        bool(getattr(frame, field))
        for field in (
            "predicate", "object", "instrument", "theme", "location", "label",
            "event_identity", "event_context",
        )
    )
    return score, populated_roles, len(frame.clause)


def _canonicalize_claim_frames(
    frames: Iterable[EvidenceFrame],
    plan: SemanticQueryPlan,
    operation: str,
    scope_tokens: set[str],
    projection_roles: tuple[str, ...],
) -> list[EvidenceFrame]:
    """Keep one best extraction per source claim and semantic measurement.

    Retrieval can surface both a full turn and a shortened memory derived from
    that same turn.  Provenance clause text is therefore not a claim identity.
    Counts retain independent statements within the same turn. Collection and
    numeric executors keep their existing identity policies.
    """
    grouped: dict[tuple[Any, ...], list[EvidenceFrame]] = defaultdict(list)
    for frame in frames:
        base = (frame.source_session_id, frame.source_turn_id, frame.status)
        if operation in {"average", "argmax", "argmin", "difference", "sum"}:
            key = base + (frame.value, frame.unit)
        elif operation in {"count", "count_distinct", "collect_distinct"}:
            key = base
        else:
            key = (frame.provenance_identity,)
        grouped[key].append(frame)
    if operation in {"count", "count_distinct"}:
        # A turn can contain several independent claims. Only overlapping
        # copies of the same source statement are duplicate retrieval views.
        claims = {}
        for source_key, candidates in grouped.items():
            retained = []
            for frame in sorted(
                candidates,
                key=lambda item: (
                    len(normalize_phrase(item.clause)),
                    _frame_selection_quality(
                        item, plan, operation, scope_tokens, projection_roles
                    ),
                ),
                reverse=True,
            ):
                clause = normalize_phrase(frame.clause)
                duplicate = next((
                    existing for existing in retained
                    if clause == normalize_phrase(existing[0].clause)
                    or (
                        len(clause.split()) >= 4
                        and (
                            normalize_phrase(existing[0].clause).startswith(clause)
                            or f" {clause} " in f" {normalize_phrase(existing[0].clause)} "
                        )
                    )
                ), None)
                if duplicate is None:
                    retained.append([frame])
                else:
                    duplicate.append(frame)
            for index, claim_frames in enumerate(retained):
                claims[source_key + (index,)] = claim_frames
        grouped = claims
    selected = [
        max(
            candidates,
            key=lambda frame: _frame_selection_quality(
                frame, plan, operation, scope_tokens, projection_roles
            ),
        )
        for candidates in grouped.values()
    ]
    return sorted(
        selected,
        key=lambda frame: (
            str(frame.source_timestamp or ""),
            str(frame.source_session_id or ""),
            str(frame.source_turn_id or ""),
            frame.frame_id,
        ),
    )


def execute_frame_operation_proof(
    context_layer: Any,
    question: str,
    plan: SemanticQueryPlan,
    ledger: Iterable[EvidenceAtom],
    *,
    effective_operation: str,
    minimum_sessions: int = 1,
    target_unit: str | None = None,
    strict_contract: bool = True,
) -> tuple[OperationProof | None, dict[str, Any]]:
    operation = normalize_text(effective_operation)
    if operation not in FRAME_SUPPORTED_OPERATIONS:
        return None, {"applicable": False, "reason": "operation_not_frame_supported"}
    frames, diagnostics = compile_evidence_frames(
        context_layer,
        question,
        plan,
        ledger,
    )
    if not diagnostics.get("applicable"):
        return None, diagnostics
    if diagnostics.get("status") != "COMPLETE":
        return OperationProof(
            status="EXTRACTION_FAILED",
            operator=plan.operator,
            answer=None,
            reason=diagnostics.get("error") or "frame_review_incomplete",
        ), diagnostics

    requested_status = plan.requested_status
    requested_actor = plan.actor or next(
        (operand.actor for operand in plan.operands if operand.actor),
        None,
    ) or "user"
    scope_tokens = _semantic_frame_scope_tokens(plan, question)
    eligible = []
    rejected = defaultdict(int)
    rejected_frame_reasons: dict[str, str] = {}
    projection_roles = tuple(plan.projection_roles)
    if not projection_roles:
        projection_roles = (
            plan.group_role if plan.group_role != "unknown" else "event",
        )
    strict_actor_binding = strict_contract and operation in {
        "collect_distinct", "count", "count_distinct",
    }
    diagnostics["execution_contract"] = {
        "operation": operation,
        "answer_role": plan.answer_role,
        "group_role": plan.group_role,
        "result_shape": plan.result_shape,
        "projection_roles": list(projection_roles),
        "normalization_warnings": list(plan.warnings),
        "requested_actor": requested_actor,
        "requested_status": requested_status,
        "scope_tokens": sorted(scope_tokens),
        "strict_actor_binding": strict_actor_binding,
    }
    if plan.operator in {"COUNT", "COUNT_DISTINCT"}:
        diagnostics["execution_contract"]["count_scope"] = count_scope_contract(
            plan, question
        )
    count_validation = []
    frames = list(frames)
    for frame_index, frame in enumerate(frames):
        if operation in {"count", "count_distinct"}:
            atom = EvidenceAtom(
                atom_id=frame.evidence_id,
                source_turn_id=frame.source_turn_id,
                source_session_id=frame.source_session_id,
                source_role=frame.source_role,
                source_speaker=frame.source_speaker,
                source_timestamp=frame.source_timestamp,
                source_quote=frame.source_quote,
                clause=frame.clause,
                clause_index=0,
                tokens=tuple(sorted(content_tokens(frame.clause))),
            )
            actor_reference = _source_actor_reference(
                atom, frame.predicate, strict_subject=True,
            )
            if actor_reference is None:
                actor_reference = frame.actor_reference
                if (
                    actor_reference == "source_speaker"
                    and normalize_text(frame.source_role) == "user"
                    and _actors_equivalent(requested_actor, "user")
                ):
                    actor_reference = "unknown"
            status = frame.status
            status_hint = derive_event_status_hint(frame.clause)
            if status_hint in {"negated", "negated_or_cancelled"}:
                status = "negated"
            elif status_hint in {"planned", "pending"}:
                status = "planned"
            if actor_reference != frame.actor_reference or status != frame.status:
                count_validation.append({
                    "frame_id": frame.frame_id,
                    "actor_before": frame.actor_reference,
                    "actor_after": actor_reference,
                    "status_before": frame.status,
                    "status_after": status,
                })
                frame = dataclasses.replace(
                    frame, actor_reference=actor_reference, status=status,
                )
                frames[frame_index] = frame
            if frame.clause.rstrip().endswith(("...", "\u2026")):
                rejected["source_clause_truncated"] += 1
                rejected_frame_reasons[frame.frame_id] = "source_clause_truncated"
                continue
        if strict_contract and not frame.relevant:
            rejected["relevance"] += 1
            rejected_frame_reasons[frame.frame_id] = "relevance"
            continue
        if frame.actor_reference == "other":
            rejected["actor"] += 1
            rejected_frame_reasons[frame.frame_id] = "actor"
            continue
        if strict_actor_binding and frame.actor_reference != "source_speaker":
            rejected["actor_unresolved"] += 1
            rejected_frame_reasons[frame.frame_id] = "actor_unresolved"
            continue
        role = normalize_text(frame.source_role)
        speaker = normalize_text(frame.source_speaker)
        if not (
            _actors_equivalent(requested_actor, role)
            or _actors_equivalent(requested_actor, speaker)
        ):
            rejected["actor"] += 1
            rejected_frame_reasons[frame.frame_id] = "source_actor_mismatch"
            continue
        if requested_status == "completed" and frame.status != "completed":
            rejected["status"] += 1
            rejected_frame_reasons[frame.frame_id] = "status"
            continue
        if requested_status == "planned" and frame.status != "planned":
            rejected["status"] += 1
            rejected_frame_reasons[frame.frame_id] = "status"
            continue
        if requested_status in {"any", "unknown"} and frame.status in {
            "planned", "negated",
        }:
            rejected["status"] += 1
            rejected_frame_reasons[frame.frame_id] = "status"
            continue
        if not _scope_contract_matches(
            scope_tokens,
            _frame_scope_tokens(frame),
        ):
            rejected["scope"] += 1
            rejected_frame_reasons[frame.frame_id] = "scope"
            continue
        if operation == "count_distinct" and not _frame_requested_identity(
            frame, plan, scope_tokens
        ):
            rejected["requested_role_missing"] += 1
            rejected_frame_reasons[frame.frame_id] = "requested_role_missing"
            continue
        if operation == "collect_distinct":
            required_roles = (
                projection_roles
                if plan.result_shape == "tuple_set"
                else (plan.group_role,)
            )
            if not _frame_projection_values(
                frame, required_roles, scope_tokens
            ):
                rejected["requested_role_missing"] += 1
                rejected_frame_reasons[frame.frame_id] = "requested_role_missing"
                continue
        if operation in {"average", "argmax", "argmin", "difference", "sum"}:
            if frame.value is None:
                rejected["measurement_missing"] += 1
                rejected_frame_reasons[frame.frame_id] = "measurement_missing"
                continue
            if plan.answer_role in {"amount", "duration"} and frame.unit is None:
                rejected["measurement_unit_missing"] += 1
                rejected_frame_reasons[frame.frame_id] = "measurement_unit_missing"
                continue
        if operation in {"temporal_earliest", "temporal_latest"}:
            if _frame_source_date(frame) is None:
                rejected["source_date_missing"] += 1
                rejected_frame_reasons[frame.frame_id] = "source_date_missing"
                continue
            if not _frame_role_value(frame, "event", scope_tokens):
                rejected["event_identity_missing"] += 1
                rejected_frame_reasons[frame.frame_id] = "event_identity_missing"
                continue
        eligible.append(frame)
    diagnostics["eligible_frame_count"] = len(eligible)
    if operation in {"count", "count_distinct"}:
        diagnostics["count_source_validation"] = count_validation
    diagnostics["rejected_counts"] = dict(rejected)
    diagnostics["frames"] = [frame.as_dict() for frame in frames]
    diagnostics["frame_audit"] = [
        {
            "frame_id": frame.frame_id,
            "evidence_id": frame.evidence_id,
            "source_turn_id": frame.source_turn_id,
            "source_session_id": frame.source_session_id,
            "decision": (
                "eligible" if frame in eligible else "rejected"
            ),
            "reason": rejected_frame_reasons.get(frame.frame_id),
            "actor_reference": frame.actor_reference,
            "status": frame.status,
            "scope_overlap": sorted(
                scope_tokens.intersection(_frame_scope_tokens(frame))
            ),
            "requested_identity": _frame_requested_identity(
                frame, plan, scope_tokens
            ),
            "projected_values": _frame_projection_values(
                frame, projection_roles, scope_tokens
            ),
            "value": frame.value,
            "unit": frame.unit,
        }
        for frame in frames
    ]
    if not eligible:
        return OperationProof(
            status="PARTIAL" if frames else "ABSENT",
            operator=plan.operator,
            answer=None,
            reason="no_eligible_grounded_frames",
        ), diagnostics

    covered_sessions = {
        frame.source_session_id
        for frame in eligible
        if frame.source_session_id is not None
    }
    if len(covered_sessions) < max(1, int(minimum_sessions or 1)):
        return OperationProof(
            status="PARTIAL",
            operator=plan.operator,
            answer=None,
            reason="minimum_session_coverage_not_met",
            calculation={"covered_session_count": len(covered_sessions)},
        ), diagnostics

    selected = _canonicalize_claim_frames(
        eligible,
        plan,
        operation,
        scope_tokens,
        projection_roles,
    )
    diagnostics["canonical_claim_count"] = len(selected)
    diagnostics["duplicate_frame_count"] = len(eligible) - len(selected)
    selected_frame_ids = {frame.frame_id for frame in selected}
    diagnostics["canonicalization"] = {
        "selected_frame_ids": sorted(selected_frame_ids),
        "dropped_duplicate_frame_ids": sorted(
            frame.frame_id
            for frame in eligible
            if frame.frame_id not in selected_frame_ids
        ),
        "claim_identity_policy": (
            "source_statement_with_overlapping_duplicate_views"
            if operation in {"count", "count_distinct"}
            else "source_session_plus_turn_plus_status"
            if operation == "collect_distinct"
            else (
                "source_session_plus_turn_plus_status_plus_value_plus_unit"
                if operation in {"average", "argmax", "argmin", "difference", "sum"}
                else "full_provenance_identity"
            )
        ),
    }

    answer = None
    calculation = {
        "operator": operation,
        "executor": "typed_provenance_frame_executor",
        "selected_frame_ids": [frame.frame_id for frame in selected],
        "covered_session_count": len(covered_sessions),
        "provenance": [frame.as_dict() for frame in selected],
    }
    proof_certificate = {
        "plan_valid": plan.valid,
        "relevance_bound": (not strict_contract) or all(
            frame.relevant for frame in selected
        ),
        "actor_bound": all(
            frame.actor_reference == "source_speaker"
            if strict_actor_binding
            else frame.actor_reference != "other"
            for frame in selected
        ),
        "status_bound": all(
            requested_status not in {"completed", "planned"}
            or frame.status == requested_status
            for frame in selected
        ),
        "scope_bound": bool(selected),
        "projection_bound": True,
        "provenance_complete": all(
            frame.source_turn_id is not None
            and frame.source_session_id is not None
            and bool(frame.clause)
            for frame in selected
        ),
        "session_coverage_met": len(covered_sessions)
        >= max(1, int(minimum_sessions or 1)),
    }
    if operation in {"temporal_earliest", "temporal_latest"}:
        dated = [(_frame_source_date(frame), frame) for frame in selected]
        dated = [(date, frame) for date, frame in dated if date is not None]
        chooser = max if operation == "temporal_latest" else min
        extreme_date = chooser(date for date, _frame in dated)
        extreme_frames = [frame for date, frame in dated if date == extreme_date]
        identities = {
            normalize_phrase(_frame_role_value(frame, "event", scope_tokens))
            for frame in extreme_frames
            if _frame_role_value(frame, "event", scope_tokens)
        }
        if len(identities) != 1:
            return OperationProof(
                status="CONFLICTING",
                operator=plan.operator,
                answer=None,
                calculation=calculation,
                reason="temporal_extreme_is_not_unique",
                executable=False,
            ), diagnostics
        chosen = extreme_frames[0]
        answer = _frame_role_value(chosen, "event", scope_tokens)
        calculation.update(
            {
                "selected_date": extreme_date.isoformat(),
                "selected_frame_id": chosen.frame_id,
                "selected_event": answer,
            }
        )
    elif operation in {"count", "count_distinct"}:
        if operation == "count_distinct":
            identities = []
            identity_seen = set()
            for frame in selected:
                identity = _frame_requested_identity(frame, plan, scope_tokens)
                normalized = normalize_phrase(identity)
                if normalized not in identity_seen:
                    identity_seen.add(normalized)
                    identities.append(identity)
            value = float(len(identities))
            calculation["distinct_values"] = identities
        else:
            value = sum(frame.quantity for frame in selected)
        answer = _format_number(value)
    elif operation in {"sum", "average", "difference", "argmax", "argmin"}:
        measured = [frame for frame in selected if frame.value is not None]
        if len(measured) != len(selected) or not measured:
            return OperationProof(
                status="PARTIAL", operator=plan.operator, answer=None,
                reason="selected_frame_lacks_grounded_value",
            ), diagnostics
        units = {frame.unit for frame in measured if frame.unit}
        normalized_target = normalize_measurement_unit(target_unit or plan.target_unit)
        if normalized_target:
            incompatible = {unit for unit in units if unit != normalized_target}
            if incompatible:
                return OperationProof(
                    status="CONFLICTING", operator=plan.operator, answer=None,
                    reason="incompatible_measurement_units",
                ), diagnostics
            unit = normalized_target
        elif len(units) <= 1:
            unit = next(iter(units), None)
        else:
            return OperationProof(
                status="CONFLICTING", operator=plan.operator, answer=None,
                reason="incompatible_measurement_units",
            ), diagnostics
        values = [float(frame.value) for frame in measured]
        if operation == "sum":
            result = sum(values)
            answer = _format_measurement(result, unit)
        elif operation == "average":
            result = sum(values) / len(values)
            answer = _format_measurement(result, unit)
        elif operation == "difference":
            if len(values) != 2:
                return OperationProof(
                    status="PARTIAL", operator=plan.operator, answer=None,
                    reason="difference_requires_two_grounded_values",
                ), diagnostics
            result = abs(values[0] - values[1])
            answer = _format_measurement(result, unit)
        else:
            chooser = max if operation == "argmax" else min
            chosen = chooser(measured, key=lambda frame: float(frame.value))
            answer = _frame_role_value(
                chosen,
                plan.group_role if plan.group_role != "unknown" else plan.answer_role,
                scope_tokens,
            )
            if not answer:
                return OperationProof(
                    status="PARTIAL", operator=plan.operator, answer=None,
                    reason="extreme_value_label_missing",
                ), diagnostics
            result = float(chosen.value)
            calculation["selected_label"] = answer
        calculation.update({"values": values, "result": result, "unit": unit})
    elif operation == "collect_distinct":
        values = []
        value_seen = set()
        for frame in selected:
            if plan.result_shape == "tuple_set":
                projected = _frame_projection_values(
                    frame, projection_roles, scope_tokens
                )
                if projected and len(projected) == 2 and projection_roles[1] == "location":
                    value = f"{projected[0]} at {projected[1]}"
                else:
                    value = " | ".join(projected or []) or None
            else:
                value = _frame_role_value(
                    frame, plan.group_role, scope_tokens
                )
            normalized = normalize_phrase(value)
            if not normalized:
                return OperationProof(
                    status="PARTIAL", operator=plan.operator, answer=None,
                    reason="requested_collection_role_missing",
                ), diagnostics
            if normalized not in value_seen:
                value_seen.add(normalized)
                values.append(value)
        answer = ", ".join(values)
        calculation["values"] = values
    else:
        return None, diagnostics

    proof_certificate["projection_bound"] = bool(answer)
    calculation["proof_certificate"] = proof_certificate
    diagnostics["proof_certificate"] = proof_certificate
    if not all(proof_certificate.values()):
        return OperationProof(
            status="PARTIAL",
            operator=plan.operator,
            answer=None,
            calculation=calculation,
            reason="proof_certificate_incomplete",
            executable=False,
        ), diagnostics
    return OperationProof(
        status="COMPLETE",
        operator=plan.operator,
        answer=answer,
        calculation=calculation,
        executable=True,
    ), diagnostics


def _word_tokens(text: str) -> list[dict[str, Any]]:
    return [
        {
            "raw": match.group(0),
            "lower": match.group(0).lower(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in re.finditer(r"[A-Za-z0-9]+", str(text or ""))
    ]


def _event_family_terms() -> dict[str, str]:
    return {
        normalize_phrase(term): family
        for family, terms in EVENT_FAMILIES.items()
        for term in terms
    }


def _requested_event_families(plan: SemanticQueryPlan, question: str) -> set[str]:
    term_to_family = _event_family_terms()
    values = [question] + [operand.text for operand in plan.operands]
    tokens = set().union(*(content_tokens(value) for value in values))
    families = {
        family
        for term, family in term_to_family.items()
        if content_tokens(term).intersection(tokens)
    }
    return families


def _requested_event_status(plan: SemanticQueryPlan, question: str) -> str | None:
    tokens = {token["lower"] for token in _word_tokens(question)}
    completed = {
        "acquired", "attended", "bought", "completed", "finished", "obtained",
        "purchased", "received", "visited",
    }
    pending = {
        "future", "intend", "intended", "need", "pending", "plan", "planned",
        "planning", "scheduled", "will",
    }
    if tokens.intersection(completed) or (
        "did" in tokens and _requested_event_families(plan, question)
    ):
        return "completed"
    if tokens.intersection(pending):
        return "pending"
    return None


def _event_topic_tokens(plan: SemanticQueryPlan) -> set[str]:
    tokens = set().union(*(content_tokens(operand.text) for operand in plan.operands))
    operation_tokens = content_tokens(" ".join(sorted(EVENT_OPERATION_TOKENS)))
    return tokens.difference(operation_tokens)


def _first_person_identity(atom: EvidenceAtom) -> str | None:
    speaker = normalize_text(atom.source_speaker)
    role = normalize_text(atom.source_role)
    if speaker and speaker not in {"user", "assistant", "speaker user", "speaker assistant"}:
        return speaker
    if role in {"user", "assistant"}:
        return role
    if speaker in {"user", "assistant"}:
        return speaker
    return None


def _event_actor(
    tokens: list[dict[str, Any]],
    event_index: int,
    atom: EvidenceAtom,
) -> str | None:
    first_person = {"i", "me", "my", "mine", "our", "ours", "we"}
    third_person = {"he", "her", "hers", "him", "his", "she", "their", "theirs", "they"}
    lower_bound = max(0, event_index - 8)
    for index in range(event_index - 1, lower_bound - 1, -1):
        token = tokens[index]
        if token["lower"] in first_person:
            return _first_person_identity(atom)
        if token["lower"] in third_person:
            return "other-pronoun"
        if token["raw"][:1].isupper() and token["lower"] not in {"a", "an", "the"}:
            names = [token["lower"]]
            cursor = index - 1
            while cursor >= lower_bound and tokens[cursor]["raw"][:1].isupper():
                names.insert(0, tokens[cursor]["lower"])
                cursor -= 1
            return " ".join(names)
    return _first_person_identity(atom) if event_index == 0 else None


def _predicate_matches(
    tokens: list[dict[str, Any]],
) -> list[tuple[int, int, str, str, str]]:
    matches = []
    lower = [token["lower"] for token in tokens]
    definitions = [
        (parts, family, "completed")
        for parts, family in COMPLETED_EVENT_PREDICATES.items()
    ] + [
        (parts, family, "pending")
        for parts, family in PENDING_EVENT_PREDICATES.items()
    ]
    definitions.sort(key=lambda item: len(item[0]), reverse=True)
    occupied = set()
    for index in range(len(tokens)):
        for parts, family, status in definitions:
            end = index + len(parts)
            if end > len(tokens) or tuple(lower[index:end]) != parts:
                continue
            positions = set(range(index, end))
            if occupied.intersection(positions):
                continue
            occupied.update(positions)
            matches.append((index, end, " ".join(parts), family, status))
            break
    return matches


def _event_object(
    tokens: list[dict[str, Any]],
    event_end: int,
) -> tuple[str, float]:
    determiners = {"a", "an", "my", "our", "the"}
    boundaries = {
        "after", "although", "and", "as", "because", "before", "but", "during",
        "for", "from", "rather", "since", "than", "that", "then", "when", "while",
        "with",
    }
    cursor = event_end
    while cursor < len(tokens) and tokens[cursor]["lower"] in determiners:
        cursor += 1
    selected = []
    for token in tokens[cursor : cursor + 12]:
        if selected and token["lower"] in boundaries:
            break
        selected.append(token["lower"])
    while selected and selected[-1] in boundaries:
        selected.pop()
    identity = normalize_phrase(" ".join(selected))
    if identity in GENERIC_EVENT_OBJECTS:
        return "", 1.0
    quantities = {
        "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
        "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
        "ten": 10.0, "couple": 2.0, "dozen": 12.0, "pair": 2.0,
    }
    quantity = 1.0
    if selected:
        if selected[0].isdigit():
            quantity = float(selected[0])
        elif selected[0] in quantities:
            quantity = quantities[selected[0]]
    return identity, quantity


def _actors_equivalent(requested: str | None, actual: str | None) -> bool:
    requested = normalize_text(requested)
    actual = normalize_text(actual)
    if not requested or not actual:
        return False
    user_aliases = {"i", "me", "mine", "my", "speaker user", "user"}
    assistant_aliases = {"assistant", "speaker assistant"}
    if requested in user_aliases:
        return actual in user_aliases
    if requested in assistant_aliases:
        return actual in assistant_aliases
    return requested == actual


def compile_event_facts(
    plan: SemanticQueryPlan,
    question: str,
    ledger: Iterable[EvidenceAtom],
) -> tuple[list[EventFact], dict[str, Any]]:
    """Compile event tuples from raw clauses using one generic event ontology."""
    requested_actor = plan.actor or next(
        (operand.actor for operand in plan.operands if operand.actor),
        None,
    )
    actor_source = "semantic_plan"
    if requested_actor is None:
        # A subjectless conversational-memory count refers to the conversation
        # owner. Named third parties remain explicit actors and are rejected.
        requested_actor = "user"
        actor_source = "conversation_owner_default"
    requested_families = _requested_event_families(plan, question)
    requested_status = _requested_event_status(plan, question)
    if requested_status is None and plan.trusted_for_legacy_routing:
        requested_status = {
            "completed": "completed", "planned": "pending",
        }.get(plan.requested_status)
    topic_tokens = _event_topic_tokens(plan)
    diagnostics = {
        "applicable": bool(
            requested_actor
            and requested_families
            and requested_status in {"completed", "pending"}
            and topic_tokens
        ),
        "requested_actor": requested_actor,
        "requested_actor_source": actor_source,
        "requested_families": sorted(requested_families),
        "requested_status": requested_status,
        "topic_tokens": sorted(topic_tokens),
        "examined_predicate_count": 0,
        "rejected_actor_count": 0,
        "rejected_family_count": 0,
        "rejected_status_count": 0,
        "rejected_topic_count": 0,
    }
    if not diagnostics["applicable"]:
        return [], diagnostics

    facts = []
    seen = set()
    for atom in ledger or []:
        tokens = _word_tokens(atom.clause)
        clause_tokens = content_tokens(atom.clause)
        topic_overlap = len(topic_tokens.intersection(clause_tokens))
        topic_coverage = topic_overlap / max(1, len(topic_tokens))
        for event_index, event_end, predicate, family, status in _predicate_matches(tokens):
            diagnostics["examined_predicate_count"] += 1
            actor = _event_actor(tokens, event_index, atom)
            if not _actors_equivalent(requested_actor, actor):
                diagnostics["rejected_actor_count"] += 1
                continue
            if family not in requested_families:
                diagnostics["rejected_family_count"] += 1
                continue
            if status != requested_status:
                diagnostics["rejected_status_count"] += 1
                continue
            if topic_overlap == 0 or topic_coverage < 0.5:
                diagnostics["rejected_topic_count"] += 1
                continue
            object_identity, quantity = _event_object(tokens, event_end)
            if not object_identity:
                continue
            provenance_identity = (
                *atom.provenance_identity,
                normalize_text(actor),
                family,
                normalize_phrase(object_identity),
            )
            if provenance_identity in seen:
                continue
            seen.add(provenance_identity)
            facts.append(
                EventFact(
                    event_id=f"EV{len(facts) + 1}",
                    actor=str(actor),
                    family=family,
                    predicate=predicate,
                    object_identity=object_identity,
                    quantity=quantity,
                    status=status,
                    topic_coverage=topic_coverage,
                    source_turn_id=atom.source_turn_id,
                    source_session_id=atom.source_session_id,
                    source_role=atom.source_role,
                    source_speaker=atom.source_speaker,
                    source_quote=atom.source_quote,
                    clause=atom.clause,
                    provenance_identity=provenance_identity,
                )
            )
    diagnostics["compiled_event_count"] = len(facts)
    diagnostics["covered_session_count"] = len(
        {fact.source_session_id for fact in facts if fact.source_session_id is not None}
    )
    return facts, diagnostics


def execute_event_count_proof(
    plan: SemanticQueryPlan,
    question: str,
    ledger: Iterable[EvidenceAtom],
    *,
    effective_operation: str,
) -> tuple[OperationProof | None, dict[str, Any]]:
    operation = normalize_text(effective_operation)
    if operation not in {"count", "count_distinct"}:
        return None, {"applicable": False, "reason": "not_event_count_operation"}
    if plan.group_role in {"instrument", "location", "theme"}:
        return None, {
            "applicable": False,
            "reason": "specific_role_requires_typed_frames",
            "counted_role": plan.group_role,
        }
    facts, diagnostics = compile_event_facts(plan, question, ledger)
    if not diagnostics.get("applicable"):
        return None, diagnostics
    if not facts:
        return OperationProof(
            status="PARTIAL",
            operator=plan.operator,
            answer=None,
            reason="no_eligible_provenance_backed_events",
        ), diagnostics

    selected = []
    seen = set()
    for fact in facts:
        if operation == "count_distinct":
            identity = (fact.family, normalize_phrase(fact.object_identity))
        else:
            identity = fact.provenance_identity
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(fact)
    value = (
        float(len(selected))
        if operation == "count_distinct"
        else sum(fact.quantity for fact in selected)
    )
    return OperationProof(
        status="COMPLETE",
        operator=plan.operator,
        answer=_format_number(value),
        calculation={
            "operator": operation,
            "executor": "deterministic_provenance_event_count",
            "selected_event_ids": [fact.event_id for fact in selected],
            "covered_session_count": len(
                {
                    fact.source_session_id
                    for fact in selected
                    if fact.source_session_id is not None
                }
            ),
            "provenance": [fact.as_dict() for fact in selected],
        },
        executable=True,
    ), diagnostics


def _actor_matches(actor: str | None, atom: EvidenceAtom) -> bool:
    if not actor:
        return True
    requested = normalize_text(actor)
    role = normalize_text(atom.source_role)
    speaker = normalize_text(atom.source_speaker)
    clause = normalize_text(atom.clause)
    if requested in {"i", "me", "my", "user", "speaker user"}:
        return role == "user" or speaker == "user"
    if requested in {"assistant", "speaker assistant"}:
        return role == "assistant" or speaker == "assistant"
    return requested == speaker or requested in clause


def _source_timestamp_temporal(atom: EvidenceAtom) -> dict[str, Any] | None:
    """Use a turn date only after semantic event binding has succeeded."""
    if atom.event_status in {"planned", "negated_or_cancelled"}:
        return None
    match = re.search(
        r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)",
        str(atom.source_timestamp or ""),
    )
    if not match:
        return None
    try:
        value = datetime.date(*(int(part) for part in match.groups()))
    except ValueError:
        return None
    return {
        "value": value.isoformat(),
        "granularity": "day",
        "interval_start": value.isoformat(),
        "interval_end": value.isoformat(),
        "direction": "source_relative",
        "basis": "bound_event_source_timestamp",
        "source_timestamp": atom.source_timestamp,
        "event_status": atom.event_status or "completed",
    }


def _entity_acronym(value: str) -> str:
    ignored = {"a", "an", "and", "at", "for", "the", "to"}
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", normalize_text(value))
        if word not in ignored
    ]
    return "".join(word[0] for word in words if word)


def _hard_entity_alias_groups(values: Iterable[str]) -> list[tuple[str, ...]]:
    """Group full names with acronyms; keep independent entities separate."""
    entities = [normalize_phrase(value) for value in values if normalize_phrase(value)]
    groups: list[set[str]] = []
    for entity in entities:
        compact = re.sub(r"[^a-z0-9]", "", entity)
        matching_group = None
        for group in groups:
            aliases = set(group)
            if any(
                compact == _entity_acronym(alias)
                or re.sub(r"[^a-z0-9]", "", alias) == _entity_acronym(entity)
                for alias in aliases
            ):
                matching_group = group
                break
        if matching_group is None:
            groups.append({entity})
        else:
            matching_group.add(entity)
    return [tuple(sorted(group)) for group in groups]


def bind_plan_operands(
    plan: SemanticQueryPlan,
    ledger: Iterable[EvidenceAtom],
    *,
    require_temporal: bool = False,
) -> tuple[dict[str, list[BoundFact]], dict[str, Any]]:
    ledger = list(ledger or [])
    bindings: dict[str, list[BoundFact]] = defaultdict(list)
    diagnostics = {"operand_diagnostics": {}, "atom_count": len(ledger)}
    for operand in plan.operands:
        operand_tokens = content_tokens(operand.text)
        hard_entity_groups = _hard_entity_alias_groups(operand.hard_entities)
        candidates = []
        for atom in ledger:
            actor_match = _actor_matches(operand.actor or plan.actor, atom)
            if not actor_match:
                continue
            clause_text = normalize_phrase(atom.clause)
            hard_match = all(
                any(alias in clause_text for alias in alias_group)
                for alias_group in hard_entity_groups
            )
            if hard_entity_groups and not hard_match:
                continue
            atom_tokens = set(atom.tokens)
            overlap = len(operand_tokens.intersection(atom_tokens))
            coverage = overlap / max(1, len(operand_tokens))
            minimum_coverage = 0.15 if hard_entity_groups else 0.34
            if overlap == 0 or coverage < minimum_coverage:
                continue
            temporal_expression = atom.temporal_expression
            if require_temporal and temporal_expression is None:
                temporal_expression = _source_timestamp_temporal(atom)
            if require_temporal and temporal_expression is None:
                continue
            hard_entity_weight = 0.30 if hard_entity_groups else 0.10
            score = min(
                1.0,
                0.58 * coverage
                + 0.17 * (overlap / max(1, len(atom_tokens)))
                + 0.15 * float(actor_match)
                + hard_entity_weight
                * float(hard_match or not hard_entity_groups),
            )
            candidates.append(
                BoundFact(
                    operand_id=operand.operand_id,
                    atom_id=atom.atom_id,
                    score=score,
                    token_coverage=coverage,
                    actor_match=actor_match,
                    hard_entity_match=hard_match or not hard_entity_groups,
                    provenance_identity=atom.provenance_identity,
                    temporal_expression=temporal_expression,
                    source_turn_id=atom.source_turn_id,
                    source_session_id=atom.source_session_id,
                    source_quote=atom.source_quote,
                    clause=atom.clause,
                )
            )
        candidates.sort(key=lambda item: (item.score, item.token_coverage), reverse=True)
        bindings[operand.operand_id] = candidates
        diagnostics["operand_diagnostics"][operand.operand_id] = {
            "candidate_count": len(candidates),
            "candidate_atom_ids": [item.atom_id for item in candidates[:8]],
            "top_score": candidates[0].score if candidates else None,
        }
    return dict(bindings), diagnostics


def select_unique_bindings(
    plan: SemanticQueryPlan,
    bindings: dict[str, list[BoundFact]],
) -> tuple[list[BoundFact], str, str | None]:
    selected = []
    missing = []
    conflicting = []
    for operand in plan.operands:
        candidates = bindings.get(operand.operand_id) or []
        if not candidates:
            missing.append(operand.operand_id)
            continue
        best = candidates[0]
        if best.score < 0.56:
            missing.append(operand.operand_id)
            continue
        near_ties = [candidate for candidate in candidates if best.score - candidate.score <= 0.08]
        semantic_values = {
            (
                normalize_text(candidate.clause),
                (candidate.temporal_expression or {}).get("value"),
            )
            for candidate in near_ties
        }
        if len(semantic_values) > 1:
            # Repeated mentions of the same dated event are safe; competing event
            # clauses or dates are not.
            dates = {
                (candidate.temporal_expression or {}).get("value")
                for candidate in near_ties
            }
            if None in dates or len(dates) != 1:
                conflicting.append(operand.operand_id)
                continue
        selected.append(best)

    if conflicting:
        return selected, "CONFLICTING", "ambiguous_operand_binding:" + ",".join(conflicting)
    if missing:
        outcome = "ABSENT" if not any(bindings.values()) else "PARTIAL"
        return selected, outcome, "missing_operand_binding:" + ",".join(missing)
    return selected, "COMPLETE", None


def _date_from_binding(binding: BoundFact) -> datetime.date | None:
    value = (binding.temporal_expression or {}).get("value")
    if not value or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        return None
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        return None


def execute_temporal_proof(plan: SemanticQueryPlan, ledger: Iterable[EvidenceAtom]) -> tuple[OperationProof, dict[str, Any]]:
    bindings, diagnostics = bind_plan_operands(plan, ledger, require_temporal=True)
    selected, outcome, reason = select_unique_bindings(plan, bindings)
    if outcome != "COMPLETE":
        return OperationProof(
            status=outcome,
            operator=plan.operator,
            answer=None,
            bindings=tuple(selected),
            reason=reason,
        ), diagnostics

    dates = [_date_from_binding(binding) for binding in selected]
    if any(value is None for value in dates):
        return OperationProof(
            status="EXTRACTION_FAILED",
            operator=plan.operator,
            answer=None,
            bindings=tuple(selected),
            reason="selected_temporal_value_lacks_day_precision",
        ), diagnostics

    if (
        plan.operator in {"ELAPSED_TIME", "DIFFERENCE"}
        and plan.answer_type in {"duration", "elapsed time", "time"}
        and len(dates) == 2
    ):
        days = abs((dates[1] - dates[0]).days)
        unit = normalize_text(plan.target_unit or "days").rstrip("s")
        if unit == "week" and days % 7 == 0:
            value, answer_unit = days // 7, "week"
        elif unit == "month":
            value = abs((dates[1].year - dates[0].year) * 12 + dates[1].month - dates[0].month)
            answer_unit = "month"
        elif unit == "year":
            months = abs((dates[1].year - dates[0].year) * 12 + dates[1].month - dates[0].month)
            if months % 12:
                return OperationProof(
                    status="CONFLICTING", operator=plan.operator, answer=None,
                    bindings=tuple(selected), reason="non_integral_requested_year_difference",
                ), diagnostics
            value, answer_unit = months // 12, "year"
        else:
            value, answer_unit = days, "day"
        suffix = answer_unit if value == 1 else answer_unit + "s"
        return OperationProof(
            status="COMPLETE",
            operator=plan.operator,
            answer=f"{value} {suffix}",
            bindings=tuple(selected),
            calculation={
                "operator": "absolute_calendar_difference",
                "left": dates[0].isoformat(),
                "right": dates[1].isoformat(),
                "result": value,
                "unit": answer_unit,
            },
            executable=True,
        ), diagnostics

    if plan.operator == "ORDER" and len(dates) >= 2:
        ordered = sorted(zip(dates, plan.operands, selected), key=lambda item: item[0])
        answer = ", then ".join(item[1].text for item in ordered)
        return OperationProof(
            status="COMPLETE", operator=plan.operator, answer=answer,
            bindings=tuple(item[2] for item in ordered),
            calculation={"operator": "chronological_sort"}, executable=True,
        ), diagnostics

    if plan.operator in {"LATEST", "EARLIEST"}:
        chooser = max if plan.operator == "LATEST" else min
        index = chooser(range(len(dates)), key=lambda item: dates[item])
        return OperationProof(
            status="COMPLETE", operator=plan.operator,
            answer=plan.operands[index].text, bindings=(selected[index],),
            calculation={"operator": plan.operator.lower(), "date": dates[index].isoformat()},
            executable=True,
        ), diagnostics

    return OperationProof(
        status="EXTRACTION_FAILED", operator=plan.operator, answer=None,
        bindings=tuple(selected), reason="operator_not_supported_by_temporal_executor",
    ), diagnostics


def compiler_state(
    plan: SemanticQueryPlan,
    ledger: Iterable[EvidenceAtom],
    proof: OperationProof | None,
) -> str:
    if not plan.valid:
        return "EXTRACTION_FAILED"
    if proof is not None:
        return proof.status if proof.status in EVIDENCE_OUTCOMES else "EXTRACTION_FAILED"
    return "ABSENT" if not list(ledger or []) else "PARTIAL"
