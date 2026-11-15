# Evidence-first State V4: Validation Failed

## Decision

Do not promote this experimental path into production benchmarking. Keep
`--pragmos-state-mode off`. The implementation and deterministic counterexamples
are complete, but the requested Knowledge-update accuracy repair is **not**.
There is no evidence here for 36-38/40, 40/40, or global zero regression.

## Implemented

- Immutable assertion addresses and exact original source/session/turn/speaker,
  timestamp and character-offset provenance.
- Grammar-constrained source spans, atomic collection members and same-source
  measurement qualification recovery. Numeral-bearing names stay distinct from
  measurements. Ambiguous bare measurements do not borrow another value's unit.
- View-local subject/possessor/entity references; separate people, objects and
  rules; strict account separation; witnessed correction replay and modality
  contracts using the existing typed executor.
- One evidence-aware request binding rather than the V3 question-only classifier
  chain. All admitted assertions remain visible, including statements omitted by
  extraction and source definitions. One bounded retry may reread up to three
  explicitly addressed omissions; it cannot relabel existing extracted claims.
- Hard 2K input/output accounting with adaptive output reservation, a 64-token
  margin, a 128-token minimum output allowance and explicit truncation rejection.
- Conservative promotion: missing or unselected evidence does not contradict an
  existing answer. Unknown paraphrases and numeral-bearing names are preserved.
  Failed candidate calls restore model state. Off and shadow do not promote.
- Batch, source-frame, omission-retry, request-budget and rejected-plan traces.

No benchmark IDs, reference answers, category labels, topic lookup tables or
question-pattern handlers were introduced into inference. No model weights,
hosted service or dependency was added. The paused NLI path remains paused.
The graph core, raw runner and datasets were not changed.

## Completed Native Check

Artifact:
`/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/knowledge_update_40_v1/runs/state_views_development/replay_evidence_v4_paired19_v8.json`

This is development replay of saved retrieval through the production candidate
boundary, **not** a new haystack ingestion run, official benchmark or judge score.
The ten repair/partial-answer targets and all nine baseline exact-correct controls
were fixed before this run. There was no best-of selection across versions.

| Measure | Result |
| --- | --- |
| Completed cases | 19/19 |
| Prior failure/partial-answer targets | 10 |
| Baseline exact-correct controls | 9 |
| Promoted repairs | 0 |
| Baseline/candidate exact matches | 9/19 -> 9/19 |
| Exact-correct controls regressed | 0/9 |
| Completed typed computations | 0 |

The JSON's `candidate_proven_count: 6` is **not six correct answers**. All six
COMPLETE results are conditional support-abstention candidates, and none was
promoted. Final predictions are identical to their saved baseline predictions.
Preserving an old wrong answer is not an improvement. Preserving nine controls
by declining every candidate does not demonstrate a successful new resolver.

Rejections: nine exhausted the bounded omission retry, one truncated extraction,
two referenced invalid/undeclared series and one lacked a source-reference-link
witness. These are failures of the experimental semantic interface, not fixes.
Earlier failed and interrupted preflights are retained and must not be pooled.

Configuration: local Phi-3 Q4, n_ctx=2048, eight threads, 40 GPU layers, seed 42;
intermediate calls use temperature=0 and top_p=1. No API judge was called.
Model, dataset, input traces, source files and configuration are hashed in the
artifact. References are read for scoring only after candidate generation.

Dataset SHA256: `65431f373ab32a77db19897639499e7fe042ea152ca46b8af71daac60adf4aa1`.
Model SHA256: `8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef`.

## Remaining Root Causes

1. Source extraction still omits decisive relations or attaches the wrong holder,
   scope or status, even when each copied string is literal source text.
2. The binder can treat property descriptions as mandatory identity anchors and
   abstain despite available evidence. Source copying is not semantic entailment.
3. Unextracted assertions are not dependency-addressed strongly enough: the model
   keeps requesting more statements instead of completing a supported binding.
4. The expression schema does not make declared series and expression references
   dependent during decoding. Invalid references are rejected, not repaired.
5. Small-model source interpretation and expression selection remain unreliable.
   Increasing context or retry limits alone does not establish correctness.

The earlier plain-reading diagnostic demonstrates that the base model can read
many of these facts. It does not validate this structured interface. The next
semantic-interface revision needs a paired native repair gate before another
full 40-question run; adding question handlers or weakening guards is not a fix.

## Regression Tests And Snapshot Limits

- Focused suite: 864 tests, 862 passed, two skipped.
- Includes 36 new evidence-first counterexample tests, model-state restoration,
  both adapters and default-off parity across six existing behaviour families.
- Full discovery: 1,116 tests; 1,110 passed, four skipped, one failure, one error.
  The error is the publication harness correctly rejecting the changed adapter
  against the old frozen tag. The failure is in the unchanged, paused semantic
  checker (`'BIRTH'` versus `plausible_interpretation_unresolved`). Neither test
  was removed, rewritten or reported as passing.

After native V8, a deterministic source-range guard was extended to recover an
upper endpoint whose preceding hyphen the numeric lexer treats as a sign.
Decimal.copy_abs preserves exact comparisons without context rounding. The test
covers both endpoints and a unit-bearing endpoint. All 150 stored source frames
from the 19-case run were rechecked: values and rejection reasons were identical
before/after this guard. No model prompt, schema or decoding decision changed.
This is deterministic replay parity, not a second native run.

Native V8 state-binding SHA256:
`0486a93dff76299094aad72ea6d58ca34a05724b82bf30fef18d1dfa90bc4ac5`.
Final state-binding SHA256:
`395f25323555dbef5331bcc95baffa2bdbd5f1f68ae93d76defb6f5cf4489cbd`.

These checks provide limited, explicit regression evidence. They do not prove
semantic equivalence, retrieval completeness, all-40 accuracy or no regression
on every question family. The accuracy acceptance gate remains failed.
