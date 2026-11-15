# Evidence reliability changes, 2026-11-14

## Scope

Implemented in `/Users/moitra/.codex/worktrees/lme-evidence-compiler/context`.
These changes cover source-preserving packing, bounded extraction recovery,
and complete answer rendering. They do not implement the deferred consolidation
of conflicting model decisions about a request.

No semantic, temporal, operation or state planner/executor module was changed.
The paused checker experiment remains unchanged. No hosted model, database or
new Python dependency is added. Raw Phi-3 answer policies and generation
configuration are unchanged. Existing uncommitted state-experiment work was
preserved, not adopted as part of this change.

## Changes

### Source-preserving context

`PRAGMOS_evidence_context.py` retains original source offsets and parses source
layout, not question patterns. Raw embedding input remains in its previous
whitespace-normalized form; separately stored quotations retain their layout.

The packer resolves a selected chunk against its original turn before expanding
to a paragraph, list or table. It preserves headings, table columns, list
numbers, speaker, role, timestamp, caption metadata and source-turn identities.
Conflicting source identity or an unlocatable quote is reported, not guessed.
Older selected quotations without a resolvable raw turn remain explicitly
identified as quote-only evidence.

Whole source units are packed under the actual tokenizer budget. Large
containers can be split into existing source units, with fair sharing between
retrieved sources. Partial and skipped units are recorded. A container is marked
complete only when all its non-whitespace source characters were shown. This is
structural coverage, not proof that all relevant real-world facts are known.
Derived graph hints retain source identity, confidence and temporal metadata.

The final prompt is repacked if necessary, rather than prefix-truncated.
The question is never silently clipped. A budget too small for the question
and wrapper raises an explicit error.

### Extraction recovery and uncertainty

Preliminary extraction records truncation, capacity and failures. Failed
preliminary extraction retains its raw source for selected-evidence recovery.
Selected evidence is processed in whole source windows rather than silently
prefix-trimmed. There are at most four windows per evidence item, and at most
two extra repair calls per question. Oversized/unprocessed spans stay in the
trace. A saturated six-fact extraction may be reviewed with a twelve-fact cap;
that review is still bounded and does not certify semantic completeness.

Truncated, malformed, empty, ungrounded and failed outputs are distinguished.
Valid grounded facts survive a failed repair. Existing usable preliminary
facts are reused. Every attempt, output limit and review status is retained.
Copied protocol headers alone do not trigger costly repair calls.

A model's completion marker reports that it reviewed a window, not that the
source contains no other relevant facts. `evidence_absence_established` remains
false; no zero or abstention is certified by failed extraction alone in these
diagnostics. Existing operation/proof, actor, mandatory-input and abstention
guards are not relaxed or replaced. Downstream semantic interpretation remains
a limitation; these changes do not solve the deferred planner-consensus issue.

### Complete answers

PRAGMOS answer instructions permit requested lists and compound answers rather
than insisting on one contiguous source span. Scalar cleanup preserves second
numeric values, ranges and recognized source qualifications. Existing
single-value cleanup and grounded qualification restoration remain tested.

The configured generation token cap is not increased. The final-generation
finish reason and output-limit flag are now visible; a 64-token generation can
still be too short for a long requested list.

### Audit and reproducibility

Both benchmark adapters serialize context-packing, extraction and generation
diagnostics. Manifests hash the new helper for PRAGMOS runs. Use a new run name
and output directory for this candidate; do not mix it into an existing run.

## Verification

- 42 new tests pass, including paired protected-answer/decision checks across
  user, assistant, updates, preference, multi-session and temporal fixtures.
- 220 focused tests pass, including both adapters and core cache tests.
- AST comparison confirmed unchanged definitions for raw answer policies,
  intent/slot/operation planners, the operation executor and abstention policy.
- `git diff --check` passes.
- Native local Phi-3, 2K context: simple fact, assistant list and table lookup
  stayed correct. A compound answer retained both original/current values.
  A late source fact changed from "I do not know." to the supported code after
  the old character clipping was removed. These are five invented direct
  source checks, not full retrieval/haystack benchmark results.
- Native extraction retained both supported facts, but Phi-3 omitted the
  completion marker. The final implementation left review completeness
  unconfirmed and did not waste a retry on its copied protocol header.

Full discovery is **not green**. Before this work it already had a publication
freeze error and a paused semantic-checker failure. The existing pre-state
snapshot test additionally rejects the intentionally changed prompts and
trace schema in six family subtests. Those strict checks were left intact;
the separate paired test verifies unchanged answers, decisions, alias state,
stored memories and graph contents on the six fixtures. This does not establish
zero regression on real data. Do not treat the current candidate as the old
frozen publication revision or overwrite that tag.

Focused verification command:

```zsh
cd /Users/moitra/.codex/worktrees/lme-evidence-compiler/context
MPLCONFIGDIR=/tmp/pragmos-matplotlib \
  /Users/moitra/surya/dev_project/context/.venv/bin/python -B -m unittest \
  test_PRAGMOS_evidence_context test_PRAGMOS_benchmark_LongMemEval \
  test_PRAGMOS_benchmark_LoCoMo test_PRAGMOS_context_layer_latency
```

No official dataset was rerun and no GPT judge was invoked. The last official
190/500 result is unchanged as a measured result. Accuracy improvement and
real-data regression safety require a paired candidate run before freezing.
