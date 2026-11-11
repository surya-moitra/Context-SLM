# Typed Computation Candidate (V4)

This is a shared, opt-in reasoning component for PRAGMOS, not a new benchmark-specific answer path. The context graph, raw Phi-3 baseline, established V3 slot protocol, and paused temporal/NLI experiments are unchanged. No additional weights, hosted services, database or dependency are required.

## Rollout Boundary

The default remains `--pragmos-composable-mode off`. The existing enabled protocol remains `slots_v3`. V4 is selected explicitly:

```text
--pragmos-evidence-compiler-mode validated
--pragmos-composable-mode shadow
--pragmos-composable-protocol typed_v4
--pragmos-temporal-contract-mode off
```

Shadow computes and records a candidate but cannot replace an answer or apply a new abstention. `candidate` permits promotion of an executable, source-backed result. It is experimental: structural validation does not prove the SLM interpreted a question or fact correctly. Do not use candidate mode for an official publication run before paired validation. Use a new run name and output directory; changed code/configuration must not resume a historical run.

## Architecture

`PRAGMOS_typed_claims.py` builds immutable claims from exact source windows. Claims retain speaker binding, source identity and offsets, entity, grouping key, event link, assertion status, quantity, calendar value and raw selection. Quantity lexing parses source syntax, not question templates. Known dimensions cannot be erased. Unknown counted nouns have separate dimensions; a rate retains its denominator until a compatible count cancels it. Calendar values preserve day/month/year precision.

`PRAGMOS_typed_computation.py` asks the already loaded local SLM for small semantic choices and exact-source literals. Python constructs the complete typed record and dependency plan. The SLM does not generate Python, SQL or an arbitrary execution program and is not asked to perform arithmetic. Supported primitives include occurrence/distinct counts, collection, sum, mean, per-entity reduction and ranking, ordered subtraction, products, quotients, relative differences, comparisons, linked attribute projection and calendar-endpoint differences. Composition uses the shared table executor, rather than separate handlers for benchmark question types.

Every numerical input is bound to a source measurement. Independent transactions are not merged merely because their values match. Identity grouping is separate from event counting. One value cannot silently fill missing independent operands; a many-to-many join or tied scalar result is rejected. A grouped binary calculation requires both inputs for every admitted group: an inner join cannot silently drop an incomplete competitor. Per-member actual prices require an equality-of-price qualification bound to the same goods and actor, not an arbitrary quote or a total divided by a count.

The working evidence view starts from question/topic overlap and expands through bound entity dependencies, not arbitrary shared words. A multiword entity requires coherent overlap; a unique explicit demonstrative may recall its head noun. A quantified rate can seek the counted member type. Measurements themselves cannot become entity anchors. Each expansion retains complete focus windows and adjacent qualifications, and reruns after unresolved antecedents are bound. The raw view and the IDs outside the working view remain auditable. This is retrieval, not a proof that omitted statements are irrelevant or that the conversation is complete.

Per-member division has a typed total/count input signature; its desired unit value cannot become its own denominator. Count/rate multiplication keeps both factors and their matching member dimensions; ordinary products retain their semantic factor descriptions. A matching counted source noun constrains the rate denominator, rather than allowing a price span to become a noun. Non-agentive properties retain an exact source subject and source attribution rather than inventing a personal relationship. Repeated-fact claims require a grounded antecedent; a repeated entity name alone cannot justify omitting a transaction.

New result contracts separate exact answers from supported lower/upper bounds and scalar answers from paired records. Explicit lower bounds can contribute to a minimum sum without treating a possible estimate as an upper bound. Collections retain each name with its measured value; conflicting values for one record are not silently deduplicated. The paired-record decision is separate from the established four-way answer-format choice, so a single winner is not treated as a list of records.

Temporal contracts distinguish ordered calendar endpoints, completed whole years/months, elapsed days and daily clock differences. Clock strings are not lexed as two counts. Source-backed event offsets are resolved after extraction with dependency provenance and cycle detection, independent of retrieval order. An explicit before/after event constraint can filter records without inventing the event's date. Unknown order is an unresolved obligation, not evidence for a zero count. Bare reporting timestamps are not event timestamps; ambiguous event bindings preserve the baseline. Day-level date lookup is supported; partial-date projections, general recurrence and unstated midnight rollover are not certified by this component.

Input-state contracts are separate from the desired output state. A remainder may be pending even though its inputs are established total capacity and completed consumption. Derived operations therefore carry permissible states per operand; they do not filter every input using the final result's state. Member reductions retain a requested member-state contract. Arithmetic uses a local Decimal context without changing process-global precision.

An explicit qualification that a mandatory input is unrecorded prevents a partial total. A different-month observation is not a zero count. The limited wrong-scope veto requires the unscoped computation to reproduce the current positive baseline answer. A missing-link veto requires the baseline itself to be the grounded intermediate rather than the requested attribute. Neither retrieval failure nor parser failure establishes global absence.

## Source-Addressed Bindings

`PRAGMOS_typed_binding.py` refines specific unresolved or inconsistent contracts, not every query field. Established V3 capabilities are delegated before refinement. Valid scalar output and owner fields are retained. A scalar collection cannot masquerade as a ranked attribute lookup; arithmetic input descriptions must be independent before extraction. A specific upstream lookup output role constrains projection, without replacing the mathematical operation with the upstream plan.

Grouped arithmetic binds literal source properties once across groups. Different synonymous source labels can map to one operand role; unresolved labels cannot be dropped. The entity/occurrence key is separate from the measured property. Comparisons bind source group order separately from input-property roles. Joins include actor identity, and every admitted group must supply both inputs. Temporal change binds dated initial/final observations within scope; duplicate or imprecise endpoints are not resolved by choosing a convenient row.

If extraction copies one entire source caption into both entity and property slots, address propagation needs a separately grounded property or occurrence anchor from the same eligible actor and allowed state. It may separate only literal word-boundary components and must reach a unique decomposition. No anchor, competing decompositions, disallowed states or a different owner cannot certify a split. Already separated addresses are unchanged. Repairs and original-to-rebound claim IDs remain in the audit.

Dates beside measurements are qualifiers, not automatic calendar-arithmetic operands. Input datatype is derived from bound claims. Scalar duration change separates the requested operator from source occurrence chronology, then computes the signed result. Numeric magnitudes never choose the subtraction direction or force a positive answer. Unestablished chronology, extra inputs and missing operands remain unresolved.

The measured-duration choice uses explicit initial/final elapsed-time expressions, not an ambiguous free-form operand name. Definite-head retrieval is enabled for ordered scalar occurrences, not all factual lookups. Count-output repairs retain literal owner references; capitalization by itself does not bind a person. Owner interpretation is independent of the rejected initial ranking/output fields, so those fields cannot silently become evidence for the corrected actor scope.

The new extraction path constrains asserted entities to the focus statement. Adjacent text is retained for reference resolution, not borrowed as an independent fact. Source relations select argument spans and derive their connecting predicate from the unchanged raw clause. Forward objects are expanded with a four-edge limit, so extracting a direct object cannot silently hide a following venue. Nearby events cannot donate their objects across clause boundaries.

A literal terminal source-object type can attest a requested single attribute head; it cannot invent geographic or other external facts. Source questions are not relationship assertions. The existing source-phase parser retains planned relations through their continuation, and existing negated-relation rejection remains in force. Linked parents with conflicting states cannot be silently collapsed into one owner/occurrence binding. These are structural constraints; semantic interpretation and source-phase parsing are still fallible.

Only the new source-addressed lookup view requires a demonstrative immediately qualifying a dependency head (for example, `that university`). A relative-clause `that` elsewhere in a passage cannot admit an unrelated named object of that type. This does not change the existing arithmetic/deictic working-view policy; full literal entity dependencies remain available in both paths.

An old turn returned by targeted retrieval is reviewed if it was not previously admitted. Identity deduplication does not discard that missing operand. A returned version of an existing turn must match its authoritative text, role, speaker and timestamp. Retrieval is read-only and bounded; missing evidence is not zero. The new source-addressed lookup path does not use a context bridge alone to veto a baseline answer: that is not proof that the requested attribute is absent. Explicit missing-input qualifications retain their existing support checks.

Each refined trace records the initial and final request, source-binding implementation hash, literal selections, binding errors, property/source-role assignments and claim-ID mappings. This is provenance and structural validation, not an independent semantic checker. The paused NLI approach remains unused, and no model weights or services were added.

## Integration And Safety

The established answer is computed first. Eligibility uses the existing label-blind semantic plan; benchmark question IDs, categories, answers and answer-session annotations are not available to the candidate. Assistant-memory, preferences, synthesis, scalar updates and completed specialized answers retain their established routes. Direct factual lookup cannot be promoted as a multi-hop lookup without a grounded relationship bridge.

V4 delegates compatible ungrouped counts, collections, membership folds and known-unit reductions to the unchanged V3 engine before V4 extraction. Delegation must also preserve the requested identity role: weekdays, participants, events and locations cannot silently become generic entity labels. Weekdays are bound to literal source weekday tokens. Delegation restores the pre-planner model state first and accounts for both planning and V3 calls. Tuple, bounded, grouped, actor-relative and temporal computations do not enter that overlapping-capability route. Established completed temporal proofs are protected. A linked attribute projection must bind its subject, predicate and object within one source assertion; an adjacent reception cannot donate its venue to another event.

The wrapper saves and restores llama.cpp state, including on failure; an external LLM cache disables the candidate. Restoration failure stops the run. It does not mutate graph or alias state. One optional read-only hybrid retrieval round can seek an absent binary operand; that work is included in query wall time. Source restoration uses the authoritative turn table, not generated graph summaries.

Speaker scope is authoritative: first-person claims belong to their source speaker; possessive-relative claims belong to the related actor and retain the narrator as provenance, not ownership. Linked-event claims inherit only a grounded parent's actor. Explicit named actors and participants in one source-bound event have separate contracts; their measurements are not relabelled as the speaker's own. Different peers' own facts or relatives cannot enter one actor's total. Multiple source speakers still require a requested conversational actor already bound to a source speaker; unresolved cross-speaker scope preserves the baseline rather than guessing.

Every call reserves output within `n_ctx`. Whole source windows are either admitted or left outside the working view, never silently truncated. The literal-extraction working view is bounded to 32 source turns and 90 focus windows; the untouched raw view is not rejected merely because it contains more conversation. The overall limit is 160 model calls, including retries and planning. Calendar spans are not decimal-precision options; explicitly requested unsupported precision is rejected rather than silently ignored. These are safety limits, not a latency promise. State snapshots and repeated small model calls add RAM/time and must be measured.

## Trace And Verification

`composable_execution_result` contains the protocol version, selected engine, semantic request, model selections, per-window decisions, binding errors, immutable claims, operand-coverage status (including per-group missing inputs), executed steps, provenance, candidate answer, supported-abstention reason, promotion/shadow status and call/token counts. `extraction_complete` means admitted windows were reviewed without unresolved decisions. `source_view_complete` and `global_retrieval_completeness_proven` are not asserted merely because this review completed. `semantic_equivalence_verified` remains false.

Tests distinguish three levels:

1. Supplied-request/claim tests verify deterministic calculation and rejection of counterexamples. They do not test the model's interpretation.
2. `PRAGMOS_probe_typed_computation.py` uses real local Phi-3 on fictional explicitly supplied evidence. This tests model-facing selection, not haystack retrieval or benchmark accuracy.
3. Paired saved-retrieval and full frozen-suite runs test actual integration. No accuracy target is established by the first two levels.

Do not equate passing executor tests with fixing every failed question. Live semantic selection remains the principal risk; keep rejected candidates and their reasons visible. Historical V3 validation reports and pinned launchers are preserved and must not be rewritten to certify changed sources.
