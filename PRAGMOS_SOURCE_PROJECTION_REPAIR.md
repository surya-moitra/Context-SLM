# Source-projection repair

Checkpoint: `5206a245533835e90b65e78743f75d528dc2ca7d`.

## Scope

The shared `run_pragmos_record` runtime now arbitrates a narrowly established
source read against a generated collection plan. The existing graph, ingestion,
retrieval, reranking, global intent routing, state selectors, temporal executors,
composable executor and model settings have not been changed. This patch does
not enable the paused temporal/NLI or state experiments.

Eligibility is based on existing source-recall/ordinal capabilities and an
operation conflict: the original route is a read, but the generated route is a
collection. Arithmetic, explicit state/time constraints and named-peer binding
are not overridden by the new capability. No question IDs, answer labels,
benchmark category numbers, place/name lookup tables or new regex question
templates enter the inference logic. Existing parsers still recognize the
supported slots and selectors. The optional `include_collection=False` view of
the answer-slot helper independently checks scalar requests without changing its
normal collection-first behavior.

## Admission And Recovery

- Values must have unambiguous original-source offsets inside the selected
  source window. Role, speaker, session and timestamp must match the raw turn.
- Explicit list selectors retain their positions; requested lists retain all
  members. A derived request is not satisfied by a constituent list item.
- Subject/property obligations must be supported locally. An adjacent request
  can scope its response, but an arbitrary same-session turn cannot.
- A color cannot borrow its subject from a parent request. A scalar relation
  must bind its subject in the same source statement, or bind the literal
  complement of an explicit first-person speech act.
- Questions, recognized negative/planned statements, unresolved selectors,
  ambiguous alternatives and unproven scope do not get a new forced answer or
  a new evidence-absence claim. The established fallback remains available.
- Candidate availability is separate from admission and selection. Existing
  anchor/actor coverage is checked before skipping work. Numeric/source and
  state-attribute vetoes remain in the final decision path. Only a selected
  projection adds protection against later temporal/composable/state promotion.

Candidate extraction retains its two-repair-call ceiling. Reordering is limited
to a missing, explicitly source-bound speech-act relation. Existing direct
answers, ambiguous candidates and other routes retain their repair ordering.
The targeted repair includes the authoritative adjacent request as untrusted
context and asks for the existing four-field protocol. It does not accept or
guess missing fields from malformed triples. Previously accepted facts remain.

After admission, obsolete collection/frame model work is skipped; graph
materialization and retrieval are not rescheduled. Decisions and literal source
spans, including the request-context provenance, are recorded under
`execution_diagnostics.source_projection`. Existing call/timing fields continue
to count actual work. The full selected evidence is not replaced with a guessed
answer or inserted back into persistent memory.

## Checks Actually Performed

- 547 selected offline tests passed, including 25 new test methods with
  parameterized capability, negative-binding and orchestration cases.
- Six-family unaffected-route fixtures were compared with the checkpoint
  adapter loaded from Git: complete non-timing traces, model-call order/state,
  vector memory, aliases and graph edges match.
- Four saved failed assistant controls now yield the correct source projections:
  ordinal item, phone number, color and complete objectives. This check uses
  saved extractions, not fresh full-haystack inference.
- Across the saved assistant30 panel, only those four saved outputs are admitted
  by the projection validator. Other saved outputs are not forcibly replaced.
- None of the latest saved multi-session20 enters the new route. Existing tests
  also replay the 14 saved positive composable model selections successfully.
- A bounded native Phi-3 Q4/2K source-extraction check on the remaining restaurant
  control recovered `assistant | recommended | Cedar House` and the source
  validator admitted `Cedar House`. It used two extraction calls (one initial,
  one repair), approximately 7.41 seconds excluding model load. This is NOT the
  full retrieval/graph/haystack benchmark and is not a latency estimate for it.
- The new candidate launcher passed zsh syntax, frozen dataset SHA-256 and full
  argument/configuration validation without starting a benchmark.

Full offline discovery was also run against both adapters. The checkpoint's
1,193 existing tests and the updated suite retain the same seven failure reports
(six old full-trace snapshot comparisons plus one paused semantic-checker case)
and one frozen-publication-reference error; six tests are skipped. No guard,
snapshot expectation or reference hash was relaxed. These outstanding checks
must not be described as a completely green repository test suite.

## Measurement Boundary And Next Gate

These checks do not establish 23/30 native semantic accuracy, unchanged wall time,
zero regression on unseen language, or any official LongMemEval/LoCoMo score.
The new source certificate is a conservative structural capability, not a
universal natural-language equivalence checker. More complex and unresolved
bindings remain on the existing path.

Run the matched assistant30 candidate with:

```zsh
caffeinate -i /bin/zsh \
  /Users/moitra/.codex/worktrees/lme-evidence-compiler/context/run_PRAGMOS_assistant_memory_projection_30.zsh
```

The launcher uses the same frozen dataset/configuration and writes into
`assistant_memory_30_v1/runs/source_projection_candidate_v1`, leaving
`baseline_pre_cross_family_v1` intact. It resumes only this new run using the
normal code/configuration manifest checks.

Review the six historical controls and all 18 previously clear successes
individually; keep the two borderline cases separate. Use actual predictions and
semantic judging for any score claim, then verify the matched multi-session20
before progressing to larger packs. The historical publication tag remains
distinct from this new development code; do not bypass its frozen-core check.
