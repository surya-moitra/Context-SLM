# Proof-first execution, stages 1-3

This is opt-in scheduling, not new question parsing or a graph optimization.
Default execution retains the previous order. Graph materialization, retrieval,
prompts, extraction limits, arithmetic rules and evidence quotas are unchanged.

## Changes

1. `run_PRAGMOS_proof_first_20.zsh` uses the frozen diagnostic20 data and candidate
   mode, not shadow. Outputs use a new `execution_latency_v2/runs/proof_first`
   folder. Old runs and datasets are preserved.
2. `prepare_compiler_sources` is shared by both scheduling policies, preserving
   order, whole-turn restoration, scope expansion and source metadata.
3. An eligible positive proof can finish before selected-evidence triple extraction,
   legacy operation compilation and answer generation. Nonpositive results use
   the legacy fallback. Identical-source/contract artifacts can be reused; only
   the deterministic baseline-dependent veto is recomputed. Changed inputs
   invalidate reuse. Call and timing accounting includes every actual attempt.

## Guards

- No early negative proof or global absence claim; unknown is not zero.
- Native model state must be restored. A restoration failure is fatal.
- Source text, session, role, speaker and timestamp must match authoritative turns.
- Numeric source qualifications and mandatory operands are checked before early
  acceptance. Unresolvable qualifications fall back instead of bypassing the veto.
- Named-speaker routing, assistant recall, preferences, evidence synthesis,
  temporal plans, ordinal recall, color/list slots and specialized update selectors
  retain the existing order. Current-membership collection keeps its old eligibility.
- State/temporal experiments and planner overlap must be off. Ablations do not
  take the shortcut. Eligibility uses capabilities, not benchmark labels or IDs.

## Measurement

An early success is an actual returned answer. No baseline is invented:
`baseline_prediction` is null and the decision records that it was not run.
Skipped stages have zero time/calls and an explicit reason. Graph and retrieval
observations are still measured normally.

Early proofs have no final answer-generation prompt, hence zero final injected
context tokens. That is NOT zero LLM work. Candidate planner/extractor token usage
remains in its audit. Do not equate final-context compression with total compute
savings. Use `elapsed_seconds` for total latency and `llm_call_counts.total` for calls.

## Verification Scope

Tests cover all 20 ordered saved source views, all 14 saved positive model-selection
sequences, protected-path call/state/graph parity, trace schema, numeric vetoes,
fallback reuse and invalidation. Saved-selection replay is not fresh inference.
A short native Phi-3 sum probe returned the supported answer and restored state;
a short distinct-count probe remained unproven and correctly declined early acceptance.
These probes are not full-haystack accuracy measurements.

The full suite retains the previously identified old snapshot/publication-ref
and paused semantic-checker failures. Their guards were not weakened.

```zsh
caffeinate -i /bin/zsh \
  /Users/moitra/.codex/worktrees/lme-evidence-compiler/context/run_PRAGMOS_proof_first_20.zsh
```

`--check-only` validates local files, the frozen-data hash and all arguments.
Resume requires an unchanged manifest. Neither 20/20 correctness nor sub-300-second
mean latency is established until the actual run is evaluated.
