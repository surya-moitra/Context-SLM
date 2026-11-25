"""Conservative linguistic repairs for rejected single-statement reads.

Syntax and a local lexicon explain an anchor, not certify semantic entailment.
This module never changes a shared query profile or supplies an answer value.
"""

from functools import lru_cache
import hashlib
from importlib import metadata, util
from pathlib import Path


PARSER_PACKAGE = "en_core_web_sm"
PROTOCOL = "local-factual-scope-v1"


@lru_cache(maxsize=1)
def resource_identity():
    result = {"protocol": PROTOCOL, "available": False, "network_at_runtime": False}
    try:
        from nltk import data
        spec = util.find_spec(PARSER_PACKAGE)
        if spec is None:
            return {**result, "reason": "syntax_pipeline_not_installed"}
        root = Path(spec.origin).parent
        digest = hashlib.sha256()
        for path in sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        lexicon = data.find('corpora/wordnet.zip')
        lexicon_hash = hashlib.sha256(Path(lexicon.zipfile.filename).read_bytes()).hexdigest()
        return {**result, "available": True, "spacy": metadata.version('spacy'),
                "pipeline": PARSER_PACKAGE, "pipeline_version": metadata.version('en-core-web-sm'),
                "pipeline_sha256": digest.hexdigest(), "nltk": metadata.version('nltk'),
                "wordnet_sha256": lexicon_hash}
    except (ImportError, LookupError, OSError, metadata.PackageNotFoundError) as exc:
        return {**result, "reason": "scope_resources_unavailable:" + type(exc).__name__}


@lru_cache(maxsize=1)
def _parser():
    import spacy
    # No transformer, embeddings, text classifier, GPU, or runtime download.
    return spacy.load(PARSER_PACKAGE, exclude=['ner'])


@lru_cache(maxsize=1024)
def _senses(lemma, pos, derived=False):
    from nltk.corpus import wordnet
    senses = set(wordnet.synsets(lemma, pos=pos))
    if derived:
        for sense in tuple(senses):
            for entry in sense.lemmas():
                senses.update(r.synset() for r in entry.derivationally_related_forms())
    return frozenset(senses)


def _lexical_match(token, other, *, derivation=False, specific_source=False):
    if token.lower_ == other.lower_ or token.lemma_.lower() == other.lemma_.lower():
        return True
    if token.pos_ == 'PROPN' or other.pos_ == 'PROPN':
        return False
    tags = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
    left_pos, right_pos = tags.get(token.pos_), tags.get(other.pos_)
    if not left_pos or not right_pos:
        return False
    left = _senses(token.lemma_.lower(), left_pos, derivation)
    right = _senses(other.lemma_.lower(), right_pos, derivation)
    # Lexical synonymy is not entity resolution: "uncle" also means pawnbroker.
    if left_pos == 'n' and not any(s.lexname() in {'noun.act', 'noun.event', 'noun.process'} for s in left):
        return False
    if left.intersection(right):
        return True
    # A source may name a more specific kind of the requested common noun.
    return bool(specific_source and left_pos == right_pos == 'n'
                and any(left.intersection(s.hypernyms()) for s in right))


def _direct_members(head):
    return [t for t in head.subtree if t == head or
            (t.head == head and t.dep_ in {'compound', 'amod', 'nummod', 'poss'})]


def _information_verb(token):
    senses = _senses(token.lemma_.lower(), 'v')
    return bool(senses) and all(s.lexname() in {'verb.communication', 'verb.cognition'} for s in senses)


class FactualScopeRepair:
    def __init__(self, question, groups, group_supported):
        self.question, self.groups, self.group_supported = question, groups, group_supported
        self.decisions = []
        self.doc = None

    def __call__(self, assertion):
        decision = {"source_text": assertion, "accepted": False, "repairs": [],
                    "semantic_certification": False, "protocol": PROTOCOL}
        self.decisions.append(decision)
        if not resource_identity()['available']:
            decision['reason'] = 'scope_resources_unavailable'
            return False
        try:
            nlp = _parser()
            if self.doc is None:
                self.doc = nlp(self.question)
            q, s = self.doc, nlp(assertion)
            # A new repair must not turn qualified/negated evidence into a fact.
            from PRAGMOS_evidence_compiler import derive_event_status_hint
            if (any(t.dep_ == 'neg' for t in s)
                    or derive_event_status_hint(assertion) in {'negated', 'planned', 'pending'}):
                decision['reason'] = 'source_is_negated_or_prospective'
                return False
            wh = [t.i for t in q if t.tag_ in {'WDT', 'WP', 'WRB', 'WP$'}]
            first_wh = min(wh, default=-1)
            comma = next((t.i for t in q if t.text == ',' and t.i < first_wh), -1)
            if comma >= 0 and any(t.pos_ == 'ADP' for t in q[:comma]):
                decision['reason'] = 'preposed_event_or_time_scope_requires_existing_contract'
                return False
            if comma >= 0 and any(t.pos_ == 'VERB' and not _information_verb(t) for t in q[:comma]):
                decision['reason'] = 'preposed_action_scope_requires_existing_contract'
                return False
            repaired = []
            for group in self.groups:
                if self.group_supported(group, assertion):
                    continue
                text = str(group.get('text') or '')
                start = self.question.lower().find(text.lower())
                if start < 0 or self.question.lower().find(text.lower(), start + 1) >= 0:
                    decision['reason'] = 'ambiguous_anchor_span'
                    return False
                span = q.char_span(start, start + len(text), alignment_mode='expand')
                tokens = [t for t in span if not t.is_punct and not t.is_space]
                if not tokens:
                    return False
                functional = (all(t.pos_ in {'ADP', 'AUX', 'DET', 'SCONJ', 'CCONJ', 'PART'} for t in tokens)
                              and all(any(c.tag_ in {'WDT', 'WP', 'WRB', 'WP$'} for c in t.subtree) for t in tokens))
                detached = (group.get('label') == 'named entity' and comma >= 0
                            and all(t.i < comma and t.pos_ in {'VERB', 'ADV'} for t in tokens)
                            and not any(t.pos_ == 'ADP' for t in q[:comma]))
                if detached:
                    # A preposed physical event is still a restriction. Only an
                    # unambiguous information-access verb can describe retrieval.
                    verbs = [t for t in tokens if t.pos_ == 'VERB']
                    detached = bool(verbs) and all(_information_verb(t) for t in verbs)
                    detached = detached and not any(t.pos_ in {'PROPN', 'NUM'} or t.dep_ == 'poss' for t in q[:comma])
                if functional:
                    kind = 'grammatical_operator_not_entity'
                elif detached:
                    kind = 'detached_request_preamble_not_entity'
                elif all(t.pos_ != 'PROPN' and any(_lexical_match(t, u, derivation=True)
                         for u in s if u.dep_ != 'neg') for t in tokens):
                    kind = 'source_lexical_relation'
                else:
                    decision.update(reason='anchor_remains_unresolved', unresolved=text)
                    return False
                repaired.append({"anchor": text, "kind": kind,
                                 "question_span": [span.start_char, span.end_char]})
            if not repaired:
                decision['reason'] = 'no_anchor_repair'
                return False
            # All actual proper names survive. Capitalization alone is not NER.
            for chunk in q.noun_chunks:
                names = [t for t in chunk if t.pos_ == 'PROPN']
                if names:
                    name = q[names[0].i:names[-1].i + 1].text
                    if name.lower() not in assertion.lower():
                        decision.update(reason='named_target_not_bound', unresolved=name)
                        return False
            targets = []
            for chunk in q.noun_chunks:
                head = chunk.root
                owned = any(t.dep_ == 'poss' for t in chunk)
                quantified = any(t.tag_ == 'WRB' for t in chunk)
                # An owned event's argument ("selection of the stove") is also mandatory.
                argument = (head.dep_ == 'pobj' and head.head.head.pos_ == 'NOUN'
                            and any(t.dep_ == 'poss' for t in head.head.head.children))
                if not (owned or quantified or argument) or head.pos_ == 'PRON':
                    continue
                members = [t for t in _direct_members(head) if t.pos_ in {'NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM'}
                           and t.tag_ not in {'WDT', 'WP', 'WRB'} and t.dep_ != 'poss'
                           and not any(c.tag_ == 'WRB' for c in t.children)]
                if any(not any(_lexical_match(t, u, derivation=t == head, specific_source=t == head)
                               for u in s) for t in members):
                    decision.update(reason='owned_event_or_argument_not_bound', unresolved=chunk.text)
                    return False
                for owner in (t for t in chunk if t.dep_ == 'poss'):
                    if owner.pos_ in {'NOUN', 'PROPN'}:
                        bound = any(c.root.lemma_ == head.lemma_ and any(
                            u.dep_ == 'poss' and u.lower_ == owner.lower_ for u in c)
                            for c in s.noun_chunks)
                    else:
                        person, number = owner.morph.get('Person'), owner.morph.get('Number')
                        bound = any(u.pos_ == 'PRON' and (
                            u.lemma_.lower() == owner.lemma_.lower() or (person in [['1'], ['2']]
                            and u.morph.get('Person') == person and u.morph.get('Number') == number)) for u in s)
                    if not bound:
                        decision.update(reason='possessor_not_bound', unresolved=owner.text)
                        return False
                targets.append(chunk.text)
            if not targets:
                decision['reason'] = 'no_independent_event_target'
                return False
            decision.update(accepted=True, reason='source_backed_linguistic_scope', repairs=repaired,
                            bound_targets=targets)
            return True
        except (ImportError, LookupError, ValueError, OSError) as exc:
            decision['reason'] = 'scope_repair_unavailable:' + type(exc).__name__
            return False


if __name__ == '__main__':
    import json
    print(json.dumps(resource_identity(), indent=2))
    if not resource_identity()['available']:
        raise SystemExit('Install requirements-factual-read.txt and the local WordNet corpus first.')
