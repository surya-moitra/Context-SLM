# Observation-Only Assistance Boundary

This frozen observer remains diagnostic only. The later opt-in answer route uses
a separate request/source interpretation protocol, not this observer's labels.
See `PRAGMOS_ASSISTANCE_INTEGRATION.md` for that integration and its limitations.
The scope below records the earlier observation-only change.

## Scope

The approved first step is implemented in the active worktree. There is no
active assistance route, circumstance-bundle collector or replacement answer.
No shared intent, retrieval, extraction, packing, graph, state/temporal executor,
source-projection guard or final-answer prompt was changed for this boundary.
`PRAGMOS_context_layer_org.py` was not edited.

Both benchmark adapters accept:

```text
--pragmos-assistance-boundary off
--pragmos-assistance-boundary observe
```

`off` is the default. It imports no boundary helper, performs no observation
call and adds no observation diagnostic to the question trace. Existing run
commands do not need to change.

`observe` works only in `pragmos_context` mode. After the existing planner has
finished, it asks the same local model to classify only the question as
`ASSIST`, `LEGACY` or `UNRESOLVED`. It does not receive benchmark category,
gold answer, retrieved facts or the model's existing answer. The frozen
`question-purpose-v1` instruction and schema are hashed in the run manifest,
along with the helper and grammar implementation files. Changing modes or
protocols cannot silently resume into the same run. Use a fresh output/run name
when changing code or enabling observation; do not remove resume guards.

## Safety And Accounting

- At most one model call, greedy decoding, 32 output tokens, and an observation
  prompt/output budget capped at 2,048 tokens or the smaller model context.
- An over-budget question is skipped, never clipped. Invalid/truncated JSON,
  duplicate keys and extra fields cannot produce an accepted observation.
- Save/load-state support and no external LLM cache are required. Preparation
  errors skip observation. Call errors restore state and leave the old answer
  path available; there are no retries. A failed state restoration is fatal.
- Phi-3 role delimiters in the question are escaped as JSON data.
- The returned report is immutable. All authority fields remain false,
  including `automatic_routing_enabled`, `context_collection_allowed` and
  `answer_change_allowed`, even for an observed `ASSIST` label.
- Reports are serialized under `execution_diagnostics.assistance_boundary`.
  Its actual model-call count is included in `llm_call_counts`, including the
  proof-first early-return path. End-to-end latency includes observation work.
  Observation has real call/state-copy/prefill overhead; it is not free.

## Validation

The native preflight found a genuine false positive on a correctly answered
assistant-memory comparison question. This implementation deliberately preserves
the raw `ASSIST` observation; it does not add question-specific rules to hide
the failure. **The boundary is not approved for automatic routing.**

Checks added include malformed-output and isolation-error tests, all three
labels across six legacy families, unchanged complete non-timing traces,
unchanged legacy call order/prompts/state, and unchanged vectors, aliases and
graph nodes/edges. A separate paired check uses the exact adapter snapshot from
immediately before this change. Pre-indexed peer conversations and proof-first
call accounting are also covered. None of these fixtures establishes semantic
benchmark accuracy.

A bounded native Phi-3 Q4/2K check made ten question-only observations from a
warmed model state. All ten restored the active token prefix, last logits and
seed, and preserved the subsequent factual completion and token usage. This
tests state isolation, not ten end-to-end benchmark answers. The known false
positive was reproduced. Mean observation elapsed time was approximately 2.25
seconds on this check, excluding initial model loading and subsequent answer
generation. Full-haystack overhead and quality were not measured here.

The full repository suite was run before and after the change. It retains the
same seven failure reports and one error: six older full-trace snapshot
comparisons, one paused semantic-checker expectation, and the frozen-publication
reference mismatch. No old assertion or reference hash was relaxed. Those
checks remain outstanding; the repository must not be described as fully green.

## Focused Tests

From the active worktree:

```zsh
cd /Users/moitra/.codex/worktrees/lme-evidence-compiler/context
PY=/Users/moitra/surya/dev_project/context/.venv/bin/python
MPLCONFIGDIR=/tmp/pragmos-matplotlib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -B -m unittest -v \
  test_PRAGMOS_assistance_boundary test_PRAGMOS_source_projection \
  test_PRAGMOS_execution test_PRAGMOS_proof_execution
```

Native verification is explicitly enabled; normal discovery does not load model
weights or start benchmarks:

```zsh
PRAGMOS_ASSISTANCE_NATIVE_MODEL=/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  "$PY" -B -m unittest -v test_PRAGMOS_assistance_boundary.NativeBoundaryTests
```

The current-before/after snapshot parity test is opt-in through
`PRAGMOS_PRE_ASSISTANCE_BASELINE`; it verifies the recorded SHA-256, rather than
accepting an arbitrary reference file. The snapshot used during implementation
was `/tmp/PRAGMOS_pre_assistance_boundary_LongMemEval.py`.

Do not infer preference accuracy improvements from these changes. Assistance
generation remains unimplemented until a sufficiently reliable response-purpose
boundary is demonstrated independently of answer quality.
