# Knowledge-update Review And Proposed Changes

Date: 2026-11-11. Analysis only: none of the inference changes proposed below has been implemented in this turn. Pack authoring, integrity tests and parser-only probes do not constitute a model benchmark.

## What The Evidence Establishes

The last official GPT-4o result remains **35/78 Knowledge-update; 190/500 overall**. Current opt-in multi-session improvements have not replaced that official measurement.

Re-running the current pure routing/selector functions on the 78 official question strings, without retrieval or model inference, gives:

| Audit | Official 78 | Historical correct 35 | Historical incorrect 43 |
| --- | ---: | ---: | ---: |
| `user_memory` intent | 57 | 26 | 31 |
| `knowledge_update` intent | 21 | 9 | 12 |
| No explicit state selector | 61 | 27 | 34 |
| Latest selector, diagnostic only | 17 | 8 | 9 |
| Deterministic history selector enabled | 0 | 0 | 0 |

The saved historical trace agrees: 61 `not_applicable` state-history results, 17 `selector_not_enabled`. In the failed 43, 31 answers came from ordinary generation, six from canonicalized generation and six from safe abstention. Their operation plans were `none` for 39 and stated-duration lookup for four.

For the 42 failed cases eligible for retrieval scoring, both gold sessions were present in final-evidence ranking within ten for **36/42 (85.7%)**. The historical correct subset was **25/30 (83.3%)** among its eligible cases. These are session-level coverage indicators, not proof that every required sentence reached the final prompt or was correctly interpreted. The abstention cases without positive gold evidence are excluded by the metric implementation.

Consequently, retrieval alone is not an adequate repair. Nor is `user_memory` inherently wrong: a current-fact question can legitimately use that source-role intent. State reasoning must be available when the question and evidence warrant it, without relying on the dataset's `question_type` label or requiring the word "now".

## Exact Code Boundaries

Line numbers refer to the reviewed working tree and may move after later edits.

| Existing location | Confirmed limitation | Intended treatment |
| --- | --- | --- |
| `PRAGMOS_benchmark_LongMemEval.py:538`, `infer_query_intent` | Update routing largely follows a small cue vocabulary; implicit-current and event-specific questions often remain ordinary recall. | Keep source-role routing unchanged. Add independently validated state-query eligibility, rather than more cue expressions or label-based routing. |
| `PRAGMOS_benchmark_LongMemEval.py:9697`, `infer_state_history_selector` | Only explicit earliest/predecessor forms can execute; latest is intentionally diagnostic. Multi-slot, best-record and event-scoped contracts are not general state plans. | Preserve this resolver as a legacy path. Add a separate semantic state contract with question-span bindings and typed output slots. |
| `PRAGMOS_benchmark_LongMemEval.py:9828`, `parse_state_history_event` | Few `my/the ... from ... to ...` forms, exact attribute-token matching, one event per row. | Do not expand the regex list. Extract literal, immutable observations from full restored raw source turns using a separate schema. |
| `PRAGMOS_benchmark_LongMemEval.py:9927`, `resolve_deterministic_state_history_answer` | Requires two sessions, report-time ordering and continuous explicit transitions. A valid new snapshot can be treated as a conflicting assertion. | Do not relax this existing path globally. Introduce a read-only timeline that distinguishes snapshots, corrections, membership mutations, references and plans. |
| `PRAGMOS_benchmark_LongMemEval.py:3256`, `evaluate_composable_candidate` | Most explicit update intents are deliberately protected; only narrow current-membership cases can enter. Removing the protection would expose old correct cases to a different arithmetic planner. | Leave the protection in place. Add the state candidate after the established baseline decision, with its own off/shadow/candidate configuration. |
| `PRAGMOS_typed_claims.py:432`, `TypedClaim` | Useful actor/source bindings and statuses exist, but `change` is essentially present/absent, not a general bitemporal attribute history. Its strict schema is used by existing grammars. | Reuse source addresses and identity validation. Introduce a separate `StateObservation` type; do not add required fields to every existing claim or grammar. |
| `PRAGMOS_typed_computation.py:747`, `compile_typed_request`; `PRAGMOS_composable_execution.py:660`, `state_fold` | Existing state fold is membership present/absent, not a general arbitrary-value timeline. Existing arithmetic and distinct/rank operations remain valuable. | Keep current membership semantics unchanged. Feed state-selected operands into existing validated arithmetic/reductions where compatible. Extend only isolated operators where a real gap is verified. |
| `PRAGMOS_benchmark_LongMemEval.py:12171`, state resolution; `:12320`, answer precedence | Earlier complete operation results can win before the history result. Promoting a new result at the wrong point could replace valid answers or bypass safety guards. | Compute and record the baseline first. Candidate promotion must preserve source-qualification and actor guards, with explicit conflict reasons and complete output coverage. |
| `PRAGMOS_context_layer_org.py:1489`, `should_supersede_existing_edge`; `:1539`, `supersede_conflicting_edges`; `:3006`, `valid_from` | Replacement uses subject/relation cues; `valid_from` uses source/report time. That cannot by itself establish event identity, functional cardinality, effective time or whether a later mention is a recollection. | No graph-core edits in the first implementation. Construct the state view from immutable raw turns, including historical evidence, rather than trusting active-edge filtering as truth. Audit graph behavior separately if a remaining failure requires it. |

`PRAGMOS_context_layer_org.py` remains the graph/memory core. Much of the current answer orchestration resides in `PRAGMOS_benchmark_LongMemEval.py` and is shared with LoCoMo. A future state capability should be a shared module called by both adapters, not a benchmark-specific answer routine.

## The Generic Repair

### 1. A Source-grounded State Contract

Add an isolated module such as `PRAGMOS_state_views.py` with a separate query and observation schema. Suggested contract: actor, entity, property, relationship/event scope, requested time selector, allowed assertion status, unit, output slots and any reduction. Each identity/property/selector choice must be linked to question wording, not merely a valid JSON enum.

Use the existing local SLM and source-span bindings to select meanings; the executor validates literal addresses and executes typed operators. Do not infer meaning from official question IDs, category labels, fixture reference operations or hand-built topic tables. No paused NLI experiment or additional model is required by this plan. Syntactically valid model output does not prove semantic correctness: unsupported or conflicting contracts remain unproven.

### 2. Immutable Observations And A State View

Each observation should retain actor, entity, property, event/relationship instance, literal value and type, report time, effective time when supported, assertion status, mutation kind and source span. Recollections and duplicate references must link to their antecedents, not create a new current state.

Group by actor + entity + property + event/relationship scope. Distinguish a functional attribute from a set-valued relation. A new favourite does not automatically delete every prior liked object; a new application does not revise the older application's approval. A present snapshot can supersede a previous snapshot only within a proven common scope. Effective chronology must come from evidence; unresolved ties, incompatible scopes and unknown ordering must not be sorted into invented certainty.

Keep all observations. Derive latest, earliest, before-event and predecessor views without mutating graph nodes or erasing raw memory. When a later message quotes an old screenshot or boarding card, its report timestamp must not make the old value current. Planned values are eligible when the question asks about a plan; they must not replace completed events for a completed-event question.

### 3. Select State Before Computation

Resolve the relevant observation or set before applying existing operations. A cumulative count is a snapshot, not an increment. Current ownership is additions minus removals, not lifetime acquisitions. Count completed event identities, not mentions. Select the latest points balance before dividing by points per night. Best performance requires a proven comparison direction; it is not universally latest, minimum or maximum.

Preserve ranges, compound records, units and all requested slots. `13-15 hours` is not `14`; `5-2` is not subtraction; initially/currently is a pair, not a single scalar. Do not let final phrase compaction turn a complete nested location into only "wardrobe".

### 4. Constrained Promotion, Not A Global Reroute

Proposed future CLI: a separately default-off `--pragmos-state-mode off|shadow|candidate`. It does not exist yet. With off, there must be no extra model calls, retrieval or mutable state changes. Shadow records the baseline and candidate but returns the baseline byte-for-byte.

Only a candidate with a complete actor/property/event binding, source-backed chronology, compatible units, satisfied authoritative qualifications and all output slots can be considered for promotion. A parser failure or missing retrieved observation is not proof of absence. Unknown or incomplete candidates preserve the baseline; a new abstention needs its own defensible source/scope rule. Preserve existing numeric-input and actor-grounding guards.

Targeted retrieval, if needed, operates on a separate read-only source view and records what it adds. It must not alter established graph ingestion, the baseline evidence order, answer suffix or global anchor policy. Keep the existing correct assistant/user and preference paths protected. A state question phrased as ordinary user recall may be eligible only through the independent source-grounded contract, not a new blanket `knowledge-update` label rule.

## Coverage And Expected Mechanisms

`T`/`F` below refers to the historical official judge label for the source profile, **not** to a measured result on the new synthetic question. The numbers are the original zero-based profile indices, not the pack's runtime indices.

| Primary profile | Selected indices | Main mechanism to validate |
| --- | --- | --- |
| Best record | 366 F, 406 T | Min/max of the same actor's completed performances, not latest mention |
| Cumulative progress | 367 F, 401 T, 432 T | Distinct additions, snapshots, completed event identities and repeated references |
| Current attribute | 368 F, 388 T, 443 F | Actor binding and complete current location/employer |
| Event-scoped value | 369 F, 405 T | Correct application/service event, not another numeric attribute or storage place |
| Routine schedule | 370 F, 374 T | Implicit-current frequency/weekday versus unrealized proposal |
| Cross-actor comparison | 371 T | Current methods of two correctly linked actors |
| Range snapshot | 372 T | Preserve interval, do not add totals |
| Membership changes | 373 F, 442 F | Set additions/removals, pre-event membership and excluded known members |
| Latest completed event | 375 F | Completed trip versus later plan or old retelling |
| Earliest state | 376 T | Original location of the same object |
| Paired states | 379 F | Initial and current slots, intermediate state and other team distractors |
| Corrected threshold | 381 F | Correction versus account balance |
| Elapsed state | 382 F, 399 T | Stated use/routine duration versus ownership or target duration |
| Normalized ratio direction | 389 F | Both numerators and denominators, bound to old/new recipes |
| Change direction | 433 T | Limit changes, not actual consumption |
| Inventory quantity | 402 F | Current stock, requested dozen unit |
| Boolean state | 404 T, 431 T | Completion and spare availability |
| Planned state | 412 T | Current plan is relevant, cancelled plan is historical |
| Predecessor state | 414 F, 419 T | Immediate prior tier and the prior person's schedule |
| Time-windowed progress | 416 F | Opening-period snapshot, not later lifetime tally |
| Derived entitlement | 421 F | Current balance then whole-unit redemption |
| Numeric change | 422 T | Bound starting and current measurements with units |
| Structured record | 424 F | Preserve wins/losses tuple, exclude future goal |
| Unsupported scope | 436-440 T, 441 F | Wrong category/role/person/location/object/composite activity |

There are 22 direct source projections, six unsupported-scope questions and 12 reference-operation cases. The projections still require state/scope selection; they are not all simple copy tasks. The remaining operations cover sets, comparisons, pairs, ratios, arithmetic, extrema and distinct events. These authoring operations are never exposed to inference.

## Regression And Success Gates

1. Run one frozen current-code baseline on these exact 40 inputs, with fixed model/configuration and the paused Temporal experiment off. Judge with the synthetic answers. Establish actual correct-answer controls; the official 35/78 measurement is a different input set and cannot serve as the synthetic baseline.
2. Implement the new pure state module and source-bound contracts behind a default-off flag. Add generated fictional tests with different names, values and ordering. Test corrections, repeated screenshots, named/related actors, conflicting attributes, plans versus completion, set removals, scalar versus range versus tuple, and unsupported scope.
3. Verify byte-identical baseline answers and unchanged calls/evidence when off and shadow. Preserve legacy resolver, typed-v3/typed-v4 arithmetic, assistant-memory, preference and raw-model behavior. A new candidate returning the same answer should not count as evidence of a repair.
4. On the frozen 40, compare every previously correct answer, not only mean F1. Inspect changed answers semantically with the same fixed judge configuration. Aim for at least 35/40 judge-correct and no loss of any baseline-correct case; if either condition fails, promotion is not approved.
5. Run the broader 200-question cross-family regression suite and remaining Knowledge-update development cases before changing defaults or making an official run. Keep evidence-level regression tests too: a passing answer can hide lost chronology or missing slots.

No static review or finite test suite can guarantee zero regressions on arbitrary future conversations. This plan limits the changed behavior, preserves a default-off baseline and requires measured paired controls before promotion. **35/40 is a target, not a result or a guarantee.** Even a perfect profile-balanced development score does not demonstrate 80% official accuracy.

## Verification Performed This Turn

- 15 pack-builder/integrity tests passed, including a full real-source preservation check over all 40 haystacks.
- All 78 official profiles are inventoried once; all 24 primary profiles are represented; all 15 historically successful primary profiles have a success-profile counterpart.
- All 40 cases have two relevant history sessions, bound reference facts, recomputed answers and valid source timestamps. Background messages outside the documented edited slots are unchanged.
- The builder verifies input hashes and refuses to overwrite different frozen artifacts. The canonical V2/R2 data are unchanged.
- No model inference, API judging, official benchmark or new inference implementation was performed.
- The wider repository already had two known failing tests from the previous turn (frozen measurement-code drift and a paused semantic-checker fixture). The new data tests do not resolve or hide those failures. A fresh complete repository suite was not run this turn.
