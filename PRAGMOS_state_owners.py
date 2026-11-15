"""Evidence-linked person identities local to a state view."""

from dataclasses import asdict, dataclass
import hashlib

from PRAGMOS_composable_execution import ContractError, normalized
from PRAGMOS_typed_claims import SELF


PRONOUNS = frozenset({"he", "him", "his", "she", "her", "hers", "they", "them", "their"})


@dataclass(frozen=True)
class OwnerWitness:
    source_id: str
    speaker: str
    left: str
    right: str
    quote: str


class OwnerIndex:
    def __init__(self):
        self.parents, self.labels, self.witnesses = {}, {}, []
        self.rule_witnesses = []
        self.metadata_links = []

    def rule(self, source, quote):
        if not quote or quote not in source.text:
            raise ContractError("state_rule_scope_requires_source_witness")
        key = "rule:" + hashlib.sha256((source.source_id + "\n" + quote).encode()).hexdigest()[:16]
        self.parents.setdefault(key, key)
        self.labels.setdefault(key, set()).add(quote)
        self.rule_witnesses.append({"id": key, "source_id": source.source_id, "quote": quote})
        return key

    def reference(self, literal, speaker, kind, *, source_id=None):
        value = normalized(literal).strip(".,;:")
        if kind == "speaker":
            if value not in SELF:
                raise ContractError("state_owner_speaker_requires_pronoun")
            key = "speaker:" + normalized(speaker)
        elif kind == "named":
            if value in SELF or value in PRONOUNS or not value:
                raise ContractError("state_named_owner_requires_name")
            key = "named:" + value.removesuffix("'s")
        elif kind == "related":
            if value in SELF or value in PRONOUNS or len(value.split()) < 2:
                raise ContractError("state_related_owner_requires_relation")
            key = "related:" + normalized(speaker) + ":" + value
        else:
            raise ContractError("state_owner_reference_unresolved")
        self.parents.setdefault(key, key)
        self.labels.setdefault(key, set()).add(literal)
        role_names = {"user", "assistant", "system", "tool", "function", "unknown"}
        if kind == "speaker" and normalized(speaker) not in role_names:
            named = self.reference(speaker, speaker, "named")
            a, b = self.root(key), self.root(named)
            if a != b:
                parent, child = sorted((a, b))
                self.parents[child] = parent
            self.metadata_links.append({"source_id": source_id, "speaker": speaker,
                "source_reference": literal, "canonical_id": self.root(key), "basis": "structured_speaker_metadata"})
        return key

    def root(self, key):
        if key not in self.parents:
            raise ContractError("state_owner_unknown_identity")
        while key != self.parents[key]:
            key = self.parents[key]
        return key

    def link(self, source, left, left_kind, right, right_kind, quote):
        if not quote or quote not in source.text or left not in quote or right not in quote:
            raise ContractError("state_owner_link_requires_source_witness")
        a = self.reference(left, source.speaker, left_kind)
        b = self.reference(right, source.speaker, right_kind)
        # Two independently named people cannot become aliases through co-mention.
        if left_kind == right_kind == "named" and normalized(left) != normalized(right):
            raise ContractError("state_owner_named_alias_requires_explicit_alias_binding")
        a, b = self.root(a), self.root(b)
        if a != b:
            parent, child = sorted((a, b))
            self.parents[child] = parent
        self.witnesses.append(OwnerWitness(source.source_id, source.speaker, left, right, quote))

    def allowed(self, contract):
        requested = set()
        has_self = any(normalized(token.strip(".,;:?!")) in SELF for token in contract.question.split())
        direct_self = any(normalized(token.strip(".,;:?!")) in {"i", "me", "we", "us", "mine", "ours"}
                          for token in contract.question.split())
        for key, labels in self.labels.items():
            if key.startswith("rule:"):
                requested.add(self.root(key))
                continue
            if key.startswith("speaker:"):
                if has_self and (contract.owner_scope == "speaker" or direct_self):
                    requested.add(self.root(key))
                continue
            subject = normalized(contract.subject_ref).removesuffix("'s").strip(".,;:")
            if subject and any(normalized(label).removesuffix("'s").strip(".,;:") == subject for label in labels):
                requested.add(self.root(key))
        if contract.owner_scope == "speaker" or direct_self:
            speakers = {self.root(key) for key in self.labels if key.startswith("speaker:")}
            if len(speakers) > 1:
                raise ContractError("state_question_speaker_ambiguous")
            requested.update(speakers)
        return requested

    def as_dict(self):
        return {"identities": [{"id": key, "canonical_id": self.root(key), "labels": sorted(labels)}
                               for key, labels in sorted(self.labels.items())],
                "links": [asdict(witness) for witness in self.witnesses], "rules": self.rule_witnesses,
                "speaker_links": self.metadata_links}


def read_owner_links(ask, source, window, assertions, owners, report):
    from PRAGMOS_grounded_computation import choice, encoded, object_schema
    from PRAGMOS_typed_claims import actor_choices
    actors = list(actor_choices(window))
    related = [ref for ref in actors if len(ref.split()) > 1 and normalized(ref.split()[0]) in SELF]
    named = [ref for ref in actors if normalized(ref.split()[0]) not in SELF and ref[:1].isupper()]
    if not related or not named:
        return
    quote = ask("Copy an explicit declaration that a named HUMAN and a relationship reference identify "
        "the SAME person. Mere co-mention, ownership of an object, and an organisation are not identity "
        "declarations. Return null if absent.\nSource: " + encoded(window),
        object_schema({"quote": choice([None, *assertions])}), 96, "extractor")["quote"]
    if quote is None:
        return
    candidates = [ref for ref in actors if ref in quote]
    schema = object_schema({"left": choice(candidates), "left_kind": choice(["speaker", "named", "related"]),
        "right": choice(candidates), "right_kind": choice(["speaker", "named", "related"])})
    row = ask("Select the two HUMAN references explicitly identified as the same person. Return their "
        "source literal references and kinds. A thing is not a person.\nIdentity declaration: " + encoded(quote),
        schema, 160, "extractor")
    try:
        if row["left"] == row["right"]:
            raise ContractError("state_owner_self_alias_not_a_link")
        owners.link(source, **row, quote=quote)
        report.setdefault("owner_link_reviews", []).append({"quote": quote, "accepted": True, **row})
    except ContractError as exc:
        report.setdefault("owner_link_reviews", []).append({"quote": quote, "accepted": False, "reason": str(exc), **row})


class EntityIndex:
    """View-local referents; subjects need not be people or have human owners."""

    def __init__(self):
        self.parents, self.mentions, self.links = {}, {}, []

    def root(self, key):
        while self.parents[key] != key:
            key = self.parents[key]
        return key

    def reference(self, literal, source, assertion_id):
        if literal is None:
            return None
        value = normalized(literal).strip(".,;:")
        if not value or literal not in source.text:
            raise ContractError("state_entity_reference_not_source_bound")
        if value in SELF:
            key = "speaker:" + normalized(source.speaker)
        elif value in PRONOUNS or value in {"it", "its", "this", "that"}:
            key = "unbound:" + source.source_id + ":" + assertion_id + ":" + value
        else:
            possessive = value.split()[0] in {"my", "our"}
            key = "entity:" + (normalized(source.speaker) + ":" if possessive else "") + value.removesuffix("'s")
        self.parents.setdefault(key, key)
        self.mentions.setdefault(key, []).append({"literal": literal, "source_id": source.source_id,
            "assertion_id": assertion_id, "speaker": source.speaker})
        role_names = {"user", "assistant", "system", "tool", "function", "unknown", ""}
        if value in SELF and normalized(source.speaker) not in role_names:
            named = "entity:" + normalized(source.speaker)
            self.parents.setdefault(named, named)
            self.parents[self.root(key)] = self.root(named)
            self.links.append({"left": key, "right": named, "basis": "structured_speaker_metadata",
                               "source_id": source.source_id})
        return key

    def bind(self, left, right, assertion):
        if left not in self.parents or right not in self.parents:
            raise ContractError("state_entity_link_unknown_reference")
        # The model supplies the reference relation, never an invented name.
        # Retain its exact witness and do not call this semantic verification.
        a, b = self.root(left), self.root(right)
        if a != b:
            parent, child = sorted((a, b))
            self.parents[child] = parent
        self.links.append({"left": left, "right": right, "assertion_id": assertion.assertion_id,
            "source_id": assertion.source.source_id, "quote": assertion.text,
            "basis": "source_reference_binding", "semantic_equivalence_verified": False})

    def as_dict(self):
        return {"mentions": [{"id": key, "canonical_id": self.root(key), "witnesses": rows}
                             for key, rows in sorted(self.mentions.items())], "links": list(self.links)}
