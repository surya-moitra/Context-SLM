# State Resolution Candidate

## Current Experimental Protocol: Evidence-first V4

State resolution remains **off by default**. Explicit `--pragmos-state-protocol
evidence_v4` selects the new read-only protocol; `bound_v3` retains the older
interface. Both adapters record that choice and hash its implementation. This
is an implementation description, not an accuracy or zero-regression claim.

1. Restore authoritative source turns and retain the existing evidence admission.
   Assign assertion IDs with exact source/session/turn, speaker, timestamps and
   character offsets. Extraction never fabricates or concatenates quotations.
2. Read at most three addressed assertions per call with their source context.
   The grammar constrains copied values to that assertion's source spans;
   contextual references have a separate source-bound vocabulary. Collection
   arrays expand into separate provenance-backed member facts. Units, ranges
   and qualifications are recovered only from the same source measurement;
   an ambiguously repeated bare number cannot acquire a guessed nearby unit.
3. Bind the original question jointly with readable source facts and assertions,
   instead of a question-only datatype/owner/operation classifier chain. Request
   phrases and expression markers are constrained to literal question spans.
   Required inputs are bound histories, not computed answer alternatives.
   Plan packing reserves output tokens and rejects an over-budget packet; it
   never silently drops facts or truncates the evidence prefix.
   The planner's output reservation is capped by the space actually remaining
   after the complete input and a 64-token safety margin. A packet leaving fewer
   than 128 output tokens is rejected. Actual budgets are recorded in the trace;
   truncated JSON is rejected, not accepted as a partial plan.
   All admitted assertions remain visible, including definitions and statements
   omitted by extraction. The binder can request at most three previously
   unextracted assertion IDs for one targeted reread, then rebind once. It cannot
   reinterpret existing facts or repeatedly retry until an answer looks right.
4. Resolve view-local people, things, organisations and rules separately from
   possessors. A mother's fact is not the narrator's fact. Changing collection
   members may share one collection, but a shared parent name cannot merge
   different bank accounts. Apply the existing witnessed report-correction
   executor within each bound history. Planned histories remain planned;
   mixed actual/planned expressions currently defer instead of coercing status.
5. Reuse typed state selection, sets, reductions, arithmetic, comparisons and
   tuple execution. Assertions with unresolved meaning block only their reviewed
   dependencies. Compare the existing answer and candidate to the executed
   contract before any promotion. Missing extracted support alone cannot replace
   a positive answer with abstention. Preserve unknown paraphrases rather than
   claiming they are contradicted.
   Unselected assertions are not automatically labelled irrelevant, and a numeral
   in a name does not certify a quantity/type contradiction.

The graph core, raw runner and datasets are unchanged. Model state is restored
after successful and rejected candidates. `shadow` never promotes an answer;
`off` makes no new model calls. The legacy protected answer paths are retained.

Native validation reports are retained under `state_views_development` with source,
model, dataset and input-trace hashes. The initial three-case V4 preflights failed
the improvement gate: free-text extraction paraphrased source fields, the first
span revision exceeded planner context budgets, and subsequent planning still
misinterpreted the request. These failed reports must not be pooled into a
best-of score. See `EVIDENCE_V4_VALIDATION.md` for the final paired check.

Literal source addresses and executable algebra do **not** prove semantic
entailment. Phi-3 still interprets holders, properties, completeness, status and
the requested expression; `semantic_equivalence_verified` remains false. A fully
reviewed admitted view does not prove retrieval completeness or absence across
the full conversation. No NLI model, hosted service or new dependency is added.
Do not promote V4 into an official benchmark based on fixture tests alone.

## Historical Protocol: Bound State Program V3

The `off|shadow|candidate` interface is unchanged and still defaults to `off`.
The current path uses one question contract, independent assertion interpretation,
source-linked owners and a typed expression executor. The sections below this
one are historical records, not current accuracy claims.

- `PRAGMOS_state_contracts.py` records the complete question and binds output
  obligations, properties, owners and time. Source column types can refine that
  contract before execution; both initial and bound contracts are recorded.
  An explicit property-equivalence binding can unify repeated references across
  times/owners. Distinct required inputs and unresolved dependencies remain.
- `PRAGMOS_state_reader.py` copies literal values, then binds source predicates,
  properties and entities independently of the question. Source speech acts are
  interpreted separately from request relevance. Uncertainty has assertion,
  property, owner and modality addresses; unrelated or planned uncertainty does
  not erase an actual observation. A required unresolved assertion blocks execution.
  Every admitted window gets a review. The existing ranked evidence admission is
  retained in `source_selection`/`source_view_scope`; this does not prove that all
  relevant turns in the original conversation were retrieved.
- `PRAGMOS_state_owners.py` keeps narrator, named and related people distinct.
  Same-person declarations need source witnesses. Two speakers' identical relative
  phrases do not become one person. Trusted named speaker metadata can bind that
  speaker's first-person references; role labels such as `user` are not names.
  Merely mentioning a company/object in a question does not request its owner.
  Numeric object rules/rates retain separate witnessed scopes rather than inventing
  a human owner; an unowned location is not a rule. Ambiguous identities stay unresolved.
- `PRAGMOS_state_program.py` compiles an expression tree over bound columns into
  deterministic state selections, sets, arithmetic, comparisons and labelled
  outputs. The planner never chooses among computed answer alternatives. Forward
  references, unused required inputs, duplicate endpoints, missing members/units,
  zero denominators and incompatible outputs are rejected. Summation requires
  source-bound independent contributions, not cumulative snapshots. Source-backed
  value aliases, rate-unit bindings and as-of conditions are audited separately.
  Grammar-bound output arity is checked independently of the proposed program.
  A scalar cannot satisfy a requested list. Exact unit conversion does not add
  range endpoints or remove source bounds/qualifications.

The production adapter passes the actual existing answer into the same validation
boundary as the new answer. A legacy operator's COMPLETE label is not a shield or
permission to overwrite an answer. Equivalent supported answers are preserved;
unresolved paraphrases are also preserved. Promotion requires a supported candidate
and an existing answer contradicted by the executed contract/output or a witnessed
wrong source scope. This includes counted-object checks: a balance in points cannot
displace an answer in free nights. Unresolved extraction does not prove absence.

Model state is restored even when a candidate fails. The core graph, raw Phi-3
runner, datasets and shared temporal implementation are not edited by this change.
Adapters and replay manifests hash all three new modules. The replay refuses to
mix source versions during a run. It reports proven, promoted and unproven cases
separately, with the latter grouped by reason.

These are binding/execution checks. The local model still interprets semantic
relationships, identities and the question; `semantic_equivalence_verified` remains
false. Passing deterministic tests is not evidence of 40/40 semantic accuracy.
Native saved-retrieval results must be reviewed separately from a fresh-haystack
run and the official-style judge. Failed/interrupted development reports are kept.

The 12-case native development replay `replay_program_v3_final12_v2.json` did not
produce any promoted repairs. Ten cases are substantive prior failures; the other
two include an already-correct nights answer and a partial location answer. This
fails the improvement gate. The completed program in the location case still
selected the wrong assertion; the baseline-preservation guard prevented promotion.
The owner/output guards described above were tightened after that replay, so its
source hashes describe the tested snapshot, not the later safety-only edits.
Do not present that replay, earlier extraction probes, or fixture tests as a
successful 40-question validation. Keep the path off for production benchmarking
until native repair and paired regression gates pass. The separate interpretation-
first binding probe improved some labels but regressed another and was not adopted.

## Previous Addition: Shared Answer Contract And Source Relations

The opt-in state path now builds one `AnswerContract` before reading source values.
It records the question's literal subject, target, complete relation, requested
output role, input type/dimension, counted noun, qualifiers, modality, time scope,
and required source operands. Grammar constraints permit only question spans and
compatible output/input types; there is no four-word property limit. The contract
is serialized in `answer_contract`, separately from the executable `contract`.

Source extraction first copies assertion-bound facts, then interprets each fact.
`SourceStateRelation` retains the literal subject, predicate, property, entity,
answer role, typed value, source qualifications and temporal witness. Both matched
and unmatched relations appear in `source_relations`. Facts enter a history only
after a query binding and type check. Source property/entity/predicate differences
cannot be merged just because values share units or were assigned the same column.
Different predicate wording currently needs an explicit source transition witness;
unresolved equivalence preserves the baseline rather than guessing an alias.
Extracted array order never creates predecessor links. Ranges and numerical
records retain their literal representation; source measurements cannot become
their own property names, while numeral-bearing property names remain valid.

Both adapters and the saved-evidence replay hash `PRAGMOS_state_contracts.py` when
the state path is enabled. Default-off parity and model-state isolation remain
required. These changes implement answer contracts and separate factual relations;
they do not establish improved 40-case semantic accuracy. General operation
composition, named-owner binding and the legacy promotion policy remain separate
work. In particular, the model can still propose an unsuitable operation family;
source values in a diagnostic extraction probe are not a final correct answer.

Five invented native Phi-3 probes at 2,048 tokens recovered the expected duration,
complete repair location, requirement value, numeric record and qualified range
in `native_contract_relations_v5.json`. This measures contract/source extraction,
not end-to-end accuracy. The requirement probe also copied a separate balance;
the history identity guard prevents combining those relations. Failed earlier
protocol probes remain available as V1-V4. No best-of benchmark score is claimed.

## Prior Revision: Source-bound State V2

The current opt-in implementation uses `PRAGMOS_state_binding.py` and
`PRAGMOS_state_claims.py` behind the same
`--pragmos-state-mode off|shadow|candidate` interface. The default remains off.
The execution description later in this document records the earlier V1
implementation and its development checks; it is not the current entry path.

V2 preserves the restored source pool and treats legacy lexical phrases as review
hints, not an AND condition every update must repeat. The state-only view uses the
existing normalized MiniLM embeddings and core retrieval threshold, with explicit
entity witnesses and registered aliases when available. Literal name matching
uses word boundaries; a pronoun alias cannot become a global entity anchor.
This admission view is not proof of full-conversation coverage, and elided
identities not resolved by the available provenance/aliases remain a limitation.
Actual source text is never rewritten in graph storage. Source histories retain
their quotes, session/turn IDs, speaker, role, timestamps and transition witnesses.
Canonical query columns are separate from literal source predicates. An implicit
property has a bound column ID and a null source predicate, not an invented label
or a number misrepresented as the property name.

The local SLM copies a property's history before interpreting the requested
operation. Generic, fictional schema demonstrations are part of the prompt; they
are not benchmark examples, and copied values must still exist in the source.
Python orders linked histories and computes typed results. The SLM selects among
executable interpretations, rather than inventing topic labels as output slots.
Explicit existing selector constraints are retained, and coordinated request
spans bind output arity. One scalar request cannot become an initial/current pair
merely because the evidence contains two historical values.

Read time is bound from the query before candidate values are shown. Past tense
alone does not authorize reading a previous state; a most-recent completed event
is still the latest record. The executable menu uses named operations, not
position-dependent numeric labels, and only the bound read selector is offered.
Explicit parsed calendar/clock operands defer to the temporal path. Historical
membership requests defer rather than silently returning the current set.

The literal numeric-projection helper is not a V2 promotion route. Promotion
requires source/owner bindings, executed state selection, complete output slots,
qualification checks and successful model-state restoration. Completed sums and
cardinality results remain protected: a scalar view alone is not a reason to
replace an aggregate. A count may be replaced by a source-stated frequency only
when the question's output kind and a literal recurrence period are bound.
An established extremum operator can project the winning numeric value rather
than an unrelated event label only when the query requests that numeric value.
A requested winning entity's name is not replaced with its measurement. Other
compatible completed results are retained.

Boolean evidence uses explicit Yes/No/not-established assertions about the named
topic and source actor, then the existing predicate executor. An unrelated or
unasserted predicate is not an explicit No. A required identity that remains
unwitnessed after registered-alias checks produces a separately audited support
abstention, not a positive certificate or proof that the underlying fact is false.

Membership uses an evidenced complete initial set plus actual additions and
disposals. Each action is bound to its transaction assertion, not a later photo
of the same object. Partial confirmations cannot reset the set. Independent
event records needing aggregation are not treated as an overwritten state.

Mistaken-report correction is distinct from genuine state change. Retraction
requires a copied source qualification, a local source-speech-act interpretation
as report error, old/new values in that qualified assertion, and compatible
history order. Accepted and rejected proposals are logged. Correction targets
the relevant latest occurrence, not every earlier appearance of the same value.
Earlier genuine states and source quotes are retained. These semantic
interpretations still depend on Phi-3; literal addresses are not entailment.

This is deliberately a conservative functional-state capability, not a claim
that all Knowledge-update operations are solved. Planned-state views, unresolved
named/related owners, unbound multi-column assertions, ambiguous recurrences and
unsupported collection contracts retain the baseline. Cross-assertion correction
joins and historical membership snapshots are not newly certified. Some compound requests
without independently bound output spans are also left to the existing path.
Scope/coreference and source-history interpretation remain model-dependent;
`semantic_equivalence_verified` remains false. A scanned selected pool is not
proof of full-conversation coverage. No NLI model, hosted planner or new model
weights were added.

All versioned native replay attempts are retained under `state_views_development`.
They must not be pooled into an artificial best-of score. The final paired check
uses twelve previously failing targets and nine baseline exact-correct controls.
Local source-operator probes separately test real changes versus report errors;
their results are not benchmark scores. Saved-retrieval checks do not replace the
future full-haystack 40-case run, fixed semantic judge or cross-family live gate.

The final atomic-claim repair check, `replay_atomic_repair21_v10.json`, completed
with all twelve development targets correct on manual source/reference review
and all nine baseline exact-correct controls preserved. Exact match is 18/21:
the remaining differences are initial/current labels, thousands punctuation and
a retained requested unit. The paired native probe V7 passed eight report-error,
five read-time and eight numeric-versus-named output checks. This is a
saved-retrieval development result, not GPT judge accuracy or a full-haystack
benchmark. Earlier failed revisions remain in the development output directory.

## Scope

`PRAGMOS_state_views.py` is a shared, read-only answer capability. Both benchmark
adapters expose `--pragmos-state-mode off|shadow|candidate`; the default is **off**.
It does not modify graph ingestion, aliases, supersession, the legacy resolvers,
the final-answer suffix, the raw Phi-3 implementation, or the datasets.

This is not an official benchmark result or a guarantee of 40/40. The last
official result is not replaced by development replay results.

## V1 Execution (Historical)

1. Restore authoritative raw turns by provenance, not generated graph predicates.
   Any unresolved source reference prevents a candidate from taking over.
2. Make a separate scope-bound source view. Composite qualifiers require the
   complete literal phrase in a source turn, not one overlapping word. The
   original retrieval order, graph and memory objects are unchanged.
3. Identify the requested output form before choosing an operation. Direct
   one-value projections use a clean raw-source prompt and must match a literal
   source quantity. This avoids asking a small model to invent an intermediate
   schema label for a fact it can already read correctly.
4. Other candidates select source assertions first, then literal actors/values
   within those assertions. Ranges, compound records and membership items must
   not be replaced by counts or units alone. The separate state executor handles
   selectors, membership mutations, comparisons and compatible-unit arithmetic.
5. Failed parsing, unresolved actors, ambiguous chronology, oversized evidence,
   incomplete operands and incompatible units retain the baseline answer. They
   are not treated as proof that the conversation lacks the requested fact.

The program supports latest/earliest/predecessor/before-value selectors,
snapshots/corrections/additions/removals/references, planned versus actual states,
paired outputs, membership collection/counting and typed numerical reductions.
An executor supporting an operation does **not** mean Phi-3 reliably constructs
its semantic inputs. Native tests, not schema validity, establish that distinction.

## V1 Safety And Limitations (Historical)

- **Off** makes no additional model/source-view calls. Paired tests compare
  answers, prompts, model calls/state, vector memory, aliases and graph contents
  against the pinned pre-state implementation in all six question families.
- **Shadow** records the candidate but cannot replace the baseline answer.
- **Candidate** is an experimental opt-in. It requires resolved source addresses,
  complete structural output and restored model state. It still needs paired
  answer-level evaluation before being enabled for a publication run.
- Existing assistant/preference/synthesis paths, complete specialized resolvers
  and authoritative missing-input guards remain protected.
- Candidate calls restore the model snapshot. Restoration failure is fatal,
  rather than allowing a contaminated benchmark to continue.
- The paused temporal/NLI experiment and composable candidate cannot be combined
  with this candidate. No new model weights, hosted service or evaluator is used.
- Question IDs, category labels, reference answers and the pack's evaluation-only
  authoring audit never enter state inference. Replayer filters/metrics are
  evaluation-side only.
- `source_bound` means literal provenance was checked, **not** semantic entailment.
  `semantic_equivalence_verified` stays false. A literal projection also records
  that chronology was not deterministically proven; state selection there is
  performed by the answering SLM.
- `source_view_complete` concerns the selected view, not the whole conversation.
  Lexical scope filtering can miss aliases/coreference. An empty view preserves
  the baseline; it does not certify absence. General safe-abstention repair is
  therefore not yet established by this path.
- The query planner can still choose the wrong mathematical operation or actor.
  Invalid bindings are rejected, but finite guards cannot guarantee semantic
  correctness. A failed candidate is not counted as an improvement.
- This paired implementation executes after the baseline, so candidate mode adds
  latency. Its planner/extractor/generator calls and elapsed time are separately
  recorded. It is not a latency optimization or an unmeasured free computation.
- Hard context/call/source/answer limits stop a candidate; no evidence or partial
  output is silently prefix-truncated and certified complete.

## V1 Development Verification (Historical)

The saved-retrieval checks use the real local GGUF at 2,048 tokens, without
re-ingesting a haystack or calling GPT. All development attempts are retained;
early attempts that failed are not pooled with later successful results.

The `replay_four_v7.json` check repaired the duration, inventory and completion
failures: `8 months`, `16 dozen eggs`, and `Yes`. It preserved the existing
`2 standard free nights` answer. These are inspected source/answer comparisons,
**not GPT-4o judge labels**. The inventory answer is not an exact lexical match
to `16 dozen`; retaining the source noun is intentional.

`replay_nine_controls_v7.json` retained **9/9** baseline exact-match answers, with
zero exact-match regressions. All nine candidates remained unproven and fell
back to the baseline. This verifies the rejection/preservation behavior; it does
not establish that the new semantic planner re-solves those nine cases.

There are **70 new tests**, including paired pipeline checks across all six
families against committed revision `18d4519`. The focused cross-family suite
runs **695 tests successfully**. These are automated correctness/parity checks,
not live benchmark accuracy measurements.

The complete 40-case paired development gate and wider live cross-family suite
are still required before enabling this candidate by default. In particular,
general operation planning, relation/actor interpretation, alias-dependent scope
binding and unsupported-scope abstention are **not established as solved** by the
three repaired examples. Do not infer a 35/40 or 40/40 score from them.

The full repository suite has two previously existing failures: the measurement
harness intentionally detects drift from its older frozen publication tag, and
the paused semantic-checker `BIRTH` fixture fails. Neither check is weakened here.

Existing context-efficiency fields describe the baseline answer prompt. They
must not be presented as the total work of candidate mode: include the separate
state call/token audit and state elapsed time when reporting that configuration.

## Running A Paired Replay

From any folder on this machine:

```zsh
caffeinate -i /bin/zsh \
  /Users/moitra/.codex/worktrees/lme-evidence-compiler/context/run_PRAGMOS_state_replay_knowledge_update.zsh
```

The runner combines the first-12 and remaining-28 saved traces and records code,
model, dataset and input-trace hashes. It refuses to overwrite a previous report.
Override `PRAGMOS_STATE_REPLAY_OUTPUT` for a new version. This replay is a
development diagnostic; it is not a substitute for the later full-haystack run
and fixed official-style semantic judge.
