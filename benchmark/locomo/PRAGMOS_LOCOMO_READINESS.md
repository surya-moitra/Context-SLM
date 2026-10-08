# PRAGMOS LoCoMo Readiness Assessment

## Adapter status

`PRAGMOS_benchmark_LoCoMo.py` now provides the benchmark-facing adapter. It
validates LoCoMo-shaped JSON, preserves both named peers as factual speakers,
normalizes session dates, indexes captions as typed child evidence, isolates
each question on a fork of a once-indexed conversation, supports resumable raw
Phi-3 and PRAGMOS runs, and emits official-compatible category scores plus
dialog/session retrieval metrics. The adapter does not use gold answers or
evidence IDs during inference.

Actor-bound grounding is enforced for named-peer questions. Raw source quotes
remain unchanged, while speaker, role, timestamp, evidence type, and parent
dialog provenance are supplied in a separate evidence envelope. Direct factual
answers require the requested actor and fact anchors in the same evidence unit;
multi-hop retrieval may retain bridge evidence, but answer-bearing facts remain
bound to the requested actor. Retrieval deduplication uses provenance identity
rather than quote text, so identical statements by different speakers remain
distinct.

Install the scorer dependency in the active project environment with:

```bash
python -m pip install -r requirements-locomo.txt
```

Validate a LoCoMo-shaped file without loading either local model:

```bash
python PRAGMOS_benchmark_LoCoMo.py \
  --data-file benchmark/locomo/synthetic_v1/pragmos_synthetic_locomo_1986_v1.json \
  --validate-only
```

## Official protocol facts

The assessment uses the public LoCoMo release at commit
`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.

- The release contains ten conversations, 19-32 sessions per conversation,
  369-689 turns per conversation, and 1,986 QA items.
- Category 1 is multi-hop. The official scorer splits comma-separated answers
  and computes partial token F1. In the pinned data, 269 of 282 questions have
  evidence in multiple sessions; evidence count reaches 19.
- Category 2 is temporal. The official inference prompt appends an instruction
  to use conversation dates and return an approximate date.
- Category 3 is open-domain. It requires commonsense or external world
  knowledge in addition to retrieved conversation evidence.
- Category 4 is single-hop and uses token F1.
- Category 5 is adversarial. The official scorer awards credit only when output
  contains `no information available` or `not mentioned`.
- The official full-context renderer appends `blip_caption` when a turn contains
  one. Across the pinned QA set, 857 questions have at least one caption-bearing
  evidence turn.

## Frozen synthetic V1 coverage

The development set under `synthetic_v1` has 1,986 questions and exactly
matches the pinned release's per-conversation session counts, turn counts, and
category counts. It uses independent fictional content and has zero exact
question-string reuse against the pinned official file.

- All 282 synthetic multi-hop questions cross sessions; evidence depth reaches
  19 turns.
- Temporal questions include relative dates and genuine cross-session chains
  spanning up to four evidence turns.
- Open-domain questions reach 17 evidence turns and a mean evidence depth of
  2.14, close to the pinned release's 2.08.
- 961 questions cite caption-bearing evidence, including 109 caption-only
  single-hop answers.
- All 446 adversarial answers are null, with 301 wrong-speaker and 145
  wrong-attribute near matches.
- The manifest records the generator, dataset, and official validation-file
  SHA-256 hashes. Synthetic V1 becomes immutable when its first PRAGMOS run
  starts.

## Implemented adapter guarantees

These are benchmark-adapter guarantees rather than benchmark-specific changes
to the context graph.

The existing LongMemEval runner is not a LoCoMo adapter and must not be pointed
at the LoCoMo-shaped JSON directly.

1. **Peer-speaker ingestion**: both speakers remain named factual subjects and
   are treated as admissible user evidence.
2. **Speaker-aware evidence envelopes**: hard query anchors can use source
   speaker metadata even when the raw quote uses `I`; raw quotes remain intact.
3. **Caption ingestion**: released `blip_caption` values become typed visual
   evidence tied to the parent dialog ID. The absent images are not simulated.
4. **Category-specific prompting and scoring**: the adapter reproduces the
   pinned category-2 date instruction, category-1 comma-aware F1, and category-5
   refusal rule.
5. **Canonical adversarial refusal**: confirmed LoCoMo abstentions use
   `No information available`, matching the official implementation.
6. **Conversation isolation and reuse**: each conversation is indexed once,
   each question runs on an isolated fork, and no QA answer or evidence label is
   used during inference.

## Implemented generalization and remaining risks

### Implemented: complete multi-session collection and set operations

PRAGMOS now plans `COLLECT_DISTINCT`, `SET_UNION`, and `SET_INTERSECTION`
operations from generic list/plural wording. Retrieval expands adaptively while
new actor-bound evidence, sessions, or values continue to appear, up to a
reported ceiling. Results preserve first-seen or chronological order, retain
per-item provenance, support per-speaker grouping, and report whether retrieval
actually saturated. Open-world queries return the supported partial set instead
of claiming completeness.

The deterministic operation layer was checked against every generated
collection, intersection, and distinct-count question using only each
question's cited evidence, including 19-turn lists. This is a component check,
not an end-to-end benchmark result; retrieval recall remains a reportable risk.

### High risk: adversarial near matches

Wrong-speaker and wrong-attribute questions deliberately retrieve highly
similar evidence. PRAGMOS has strong anchor and attribute safeguards, but they
must operate on named peer speakers. Without the adapter requirements above,
the model will either abstain on valid named-speaker questions or answer invalid
ones from the other speaker's turn.

### Implemented: source-relative temporal resolution

Temporal expressions now carry value, granularity, interval, direction, source
timestamp, and event status. The resolver covers explicit day/month/year
expressions, `last week`, `next month`, weekdays, yesterday/tomorrow, stated
durations, and `N`-unit offsets without inventing day precision. Provenance-
backed relative event chains can compose across sessions, and planned events are
kept distinct from completed events. Deterministic answers require a unique
event/date binding; ambiguous cases remain on the existing fallback path.

All generated temporal families passed the same gold-evidence component audit,
including four-turn cross-session chains. End-to-end quality still depends on
retrieving every link in the chain.

### Medium risk: image-caption evidence

Ignoring captions makes some questions impossible. Treating captions as normal
untyped text can also contaminate speaker attribution. Caption evidence needs a
clear marker and the parent turn's provenance.

### Implemented routing, model-limited conclusion: open-domain inference

PRAGMOS now detects generic premise-based inference language independently of
LoCoMo category IDs. It retrieves actor-bound evidence across sessions,
compacts exact factual sentences with provenance under the final token budget,
and delegates only the conclusion to the base SLM. Inferred answer attributes
are not required to occur verbatim in premise text, so evidence such as a city
can support a country inference without weakening speaker or project scope.

All 96 synthetic open-domain questions route through this path and produce
actor-bound premises in a gold-evidence component audit. The final conclusion
remains model-limited: report category 3 separately, and do not disguise base-
model knowledge as a memory-layer failure or add benchmark-specific answer
tables.

### Lower risk: direct single-hop recall

After speaker-aware ingestion, direct category-4 facts should be PRAGMOS's
strongest LoCoMo category. Remaining risks are long causal answers, exact list
phrasing, and caption-only details.

## Publication-safe sequence

1. Implement and test a LoCoMo adapter and official-compatible scorer without
   changing PRAGMOS retrieval behavior.
2. Commit and tag the synthetic dataset, generator, adapter, scorer, model hash,
   prompt, seed, and configuration.
3. Run small fixed slices from every synthetic family, then the frozen full
   synthetic V1 exactly once for final development validation.
4. Make only correctness fixes justified by family-level failures. Preserve all
   prior synthetic results and increment the dataset or code version.
5. Freeze code and configuration. Record the Git commit before opening official
   answers or running official evaluation.
6. Run raw Phi-3 and PRAGMOS on all ten official conversations with identical
   model and generation settings. Report overall and per-category F1, retrieval
   recall using evidence IDs, latency, context tokens, and ablations.
