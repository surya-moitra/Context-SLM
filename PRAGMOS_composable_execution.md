# Source-Backed Composable Execution

This is an opt-in reasoning component shared by the LongMemEval and LoCoMo adapters. It is not a replacement for the context graph or a new retrieval system. It uses the already loaded local SLM, with no new model weights, service, database, or dependency.

## Activation

Add these flags to a new PRAGMOS run, with a new output directory and run name:

```text
--pragmos-evidence-compiler-mode validated
--pragmos-composable-mode candidate
--pragmos-temporal-contract-mode off
```

The default is `--pragmos-composable-mode off`. Off executes the established path without new planning, source restoration, model-state snapshots, or extraction calls. Do not resume an old run after changing code/configuration. Both adapters fingerprint the new modules when the candidate is enabled.

The active files are under `/Users/moitra/.codex/worktrees/lme-evidence-compiler/context`, not the older inference copies in the main project folder. The separate, pinned validation launcher is under `/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2/composable_execution_candidate_v1`.

## What Changed

- `PRAGMOS_composable_execution.py`: immutable source/fact/quantity records; raw-value grounding; dimensional conversion; scalar/collection contracts; bounded table programs; provenance-backed results.
- `PRAGMOS_grounded_computation.py`: a small reduction request, independent statement selection, exact-source argument slots, and deterministic compilation to the table executor. No benchmark labels, question IDs, or gold annotations are accepted.
- `PRAGMOS_composable_candidate.py`: native grammar validation, bounded local generation, and model-state isolation. The earlier expression-tree interface remains available for offline executor tests; it is not the integrated planner.
- `PRAGMOS_evidence_compiler.py`: one additive, read-only adapter from selected evidence references to authoritative turns. Existing executors and their defaults are unchanged.
- Benchmark adapters: explicit opt-in control, source hashes, candidate diagnostics, added-call accounting, and answer promotion. Existing prompts, retrieval, context budgets, graph updates, scoring, and paused temporal/NLI code are unchanged.

## Execution Contract

Programs consist of at most 16 dependency-ordered steps and depth 8. Operators are `source`, `filter`, `project`, `distinct`, `reduce`, `join`, `union`, `intersection`, `arithmetic`, `calculate`, `rank`, and `state_fold`. There is no generated Python, arbitrary SQL, `eval`, question-template matcher, or city/career/domain lookup table.

The integrated version-3 model-facing contract asks only for `count_distinct`, `sum`, `mean`, `collect`, or `unsupported`. It no longer asks Phi-3 to write an expression tree, choose arbitrary columns, or emit a full execution program. Python compiles the supported reduction to typed operations. More elaborate table programs are covered by supplied-plan unit tests, not claimed as capabilities of this smaller live planner.

The existing semantic planner's answer-role and result-shape contracts constrain the candidate. An amount cannot replace a requested place/person/item, and a collection cannot replace a scalar. Unsupported shapes preserve the baseline. This is a type-consistency safeguard, not a question-template classifier or semantic/NLI checker.

For instance, the executor can represent grouped spending as `source -> reduce(keys=[location], aggregate=sum) -> rank(max) -> project location`. That demonstrates executor composition, not reliable live generation of this plan. The integrated version-3 path currently certifies only its supported count, collection, sum, mean, and current-membership computations.

Facts retain their actor, predicate, roles, event/claim identity, assertion kind, status, raw measurement, canonical measurement, conversion factor, and full source envelope. The extraction prompt uses opaque source IDs rather than internal turn/session identifiers; those original identifiers remain in the provenance trace. Entity grouping does not supersede transactions. Equal amounts from independent events survive; overlapping copies of one source slot are deduplicated. Explicit numeric distinct/group/join keys use canonical numeric equality, not decimal spelling or the original unit. Optional aliases require explicit, role-scoped source bindings; a dictionary of guessed aliases is rejected.

Authoritative turns retain retrieval priority rather than being reordered chronologically. The existing declared topic contract supplies scope seeds; retrieved turns mentioning a selected exact identity can enter one additional closure pass. Within that scope, every sentence is reviewed independently. The model selects entity and actor literals from the raw sentence and a measurement index from a source-derived list. It does not retype quotes, invent quantities, or decode internal session IDs. Semicolon qualifications remain attached to their sentence. A uniquely stated explicit measure can be retained when the extractor returns null; multiple measures require an explicit selection.

Invalid slots are retried locally, at most twice. Unresolved required units prevent certification rather than producing a partial total. Explicitly different actors cannot contribute to the requested actor's result. First-person agents and possessors bind through the source speaker. This still depends on model relevance/semantic labels; literal grounding does not prove those labels are correct.

For an eligible current-set count, `state_fold` applies provenance-backed present/absent assertions in source order before counting. Repeated references are idempotent. A narrowly scoped source-assertion repair labels unresolved membership effects without answering the question. Scalar preference/address updates retain their established path. An evidenced empty set can yield zero; absent evidence cannot.

Grammar literals escape the JSON and GBNF layers independently, preserving quotes, backslashes, control characters, and Unicode in raw source spans. The candidate uses a local converter subclass, without patching the installed library or changing legacy decoders. Before a grammar reaches generation, a temporary native sampler validates it with the loaded model vocabulary and is freed immediately. Construction or native validation failure preserves the existing answer rather than attempting unconstrained generation. These checks add no model inference calls.

Every required operand must have eligible facts. A scalar must resolve uniquely; collections may contain several facts. Missing inputs are not zero. One measurement cannot stand in for two independent arithmetic roles. Ambiguous joins, tied scalar extrema, mixed dimensions/currencies, unknown conversions, invalid quantities, and unresolved source references cannot produce an executable result. Calendar months/years are not silently converted into fixed-duration days. Source ranges/bounds that need an interval contract remain unsupported rather than being certified as exact values.

## Integration And Isolation

The existing answer is computed first. Candidate eligibility uses existing label-blind semantic computation plans, never LongMemEval question types or LoCoMo category numbers. Factual LOOKUP, assistant-memory, scalar updates, preference, synthesis, selected specialized answers, confirmed source-input vetoes, and unrelated safety abstentions remain protected. A latest-selector set count may opt in only when it has no scalar requested attribute or enabled deterministic state answer. Current calendar-temporal routing and the paused NLI experiment are not replaced or re-enabled.

An eligible candidate selects source-linked facts and compiles a supported reduction. Missing reviews, invalid records, parser failure, context overflow, and length-truncated output prevent promotion. Each call is checked against `n_ctx`, including reserved output. A source statement is never silently shortened. The path is bounded to 64 model calls and 256 statement units; an answer exceeding `max_tokens` is not truncated or promoted.

Only complete source-backed results are promoted. Otherwise the old answer remains the old answer: it is not relabeled as a successful composable proof. This does not retroactively certify or repair every legacy result. The candidate never turns extraction failure into a new global abstention.

Before candidate calls, the component saves llama.cpp state and restores it in `finally`, including failure paths. It makes no embedding/reranker calls and does not mutate the graph or alias registry. Models lacking state save/restore, or using an external LLM cache, skip the candidate. A restoration failure stops execution rather than silently contaminating later questions. This uses the installed llama-cpp-python state API; it does not load a second LLM. State copies temporarily increase RAM and CPU work, which must be measured rather than assumed negligible.

## Trace And Measurement

Enabled runs add `composable_execution_result` and `composable_execution_seconds`. The result contains the request, scope seeds/identity expansion, per-unit reviews, selection errors, program, extraction completeness, step inputs/outputs, raw and normalized provenance, call/token counts, snapshot bytes, restoration status, and promotion flag. `baseline_prediction` remains available, but a separate off-mode invocation is still required for a controlled measured comparison. `extraction_complete` means the admitted scope was reviewed without unresolved units; it is not proof of exhaustive conversational retrieval.

`llm_call_counts.total` includes the candidate planner and extractor calls. Record wall time includes planning, extraction, state copying/restoration and execution. LoCoMo's query wall time includes the same work. Existing final-answer context-efficiency metrics retain their prior definition; candidate internal prompt/completion tokens are reported separately, not hidden inside the final context figure.

## Validation And Remaining Limits

The tests include grouped-versus-individual extrema, source-backed conversions, scalar/collection cardinality, actor swaps, statuses, alias provenance, duplicate versus distinct transactions, per-entity derived comparisons, role-bound tuples, missing inputs, zero denominators, dimension conservation, malformed/truncated extraction, state restoration, label isolation, both adapters' manifests, and paired default-off pipeline checks against the pre-change source snapshot.

The paired fixture test can be repeated while the temporary snapshot exists:

```zsh
cd /Users/moitra/.codex/worktrees/lme-evidence-compiler/context
MPLCONFIGDIR=/tmp/pragmos-matplotlib \
PRAGMOS_PRE_COMPOSABLE_BASELINE=/tmp/pragmos-pre-composable-20261108/PRAGMOS_benchmark_LongMemEval.py \
/Users/moitra/surya/dev_project/context/.venv/bin/python -B -m unittest \
  test_PRAGMOS_composable_execution \
  test_PRAGMOS_composable_candidate \
  test_PRAGMOS_composable_integration
```

Executor tests supply correct plans and extraction fixtures. They are not evidence that Phi-3 will generate every correct program or actor/operand assignment. Structural/type/source checks do not prove semantic equivalence; the trace explicitly records `semantic_equivalence_verified=false`. Reviewing all selected sources is not proof that retrieval found every relevant fact in the conversation. No 42/50, 110/133, or official accuracy is established by these tests.

Compare off versus candidate on the same frozen R2 questions/configuration, first a small paired smoke, then 50, then broader regression/133. Do not compare revised R2 answers with R1 golds to claim a code gain. The existing frozen-publication-reference test and paused semantic-checker fixture remain visible failures; neither is suppressed or silently rewritten for this candidate.

Small live Phi-3 checks (not benchmark scores) are saved in the separate candidate validation folder. Grouped spending and mixed hours/minutes succeeded with a nonempty saved model state restored. An individual-largest-purchase counterexample exposed a wrong output projection; with the parent contract applied, the unresolved/nonunique candidate was rejected rather than replacing the established answer. These are development checks, not an independent holdout or a promise of official accuracy.

The historical `run_candidate_smoke_grammar_fix_v2.zsh` is pinned to the previous code and must not be used for this revision. Old reports and launchers remain unchanged; do not resume their runs with changed code. Native parser tests run with `PRAGMOS_GRAMMAR_TEST_MODEL` pointing to the GGUF and load vocabulary only.

`PRAGMOS_replay_composable.py --verify` runs real Phi-3 planning and extraction on saved retrieval, then checks the baseline errors and retained exact answers. Gold answers are used only after inference for evaluation, never for selecting sources or generating facts. Reports identify the model, data, saved trace, and inference-source hashes. This is a saved-evidence development replay, not an independent holdout, GPT judge run, or end-to-end haystack benchmark.
