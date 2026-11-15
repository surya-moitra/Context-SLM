# Execution Optimization V1

Changes are in the `lme-evidence-compiler/context` worktree, not the primary
project's older copy of the inference modules. No datasets were changed.

## Scope

Default execution remains `reference`. Opt in with:

```zsh
--pragmos-execution-policy optimized
--pragmos-retrieval-device cpu
```

- Bounded exact-text token-count cache: 256 entries, at most 8192 characters per
  entry, scoped to the model identity. Failed tokenization is not cached.
- Cached per-document BM25 term counts. Token mutations invalidate the entry;
  corpus changes still update document frequencies and length normalization.
  Reset/fork isolation is retained.
- Batched independent operation/temporal retrieval queries: one embedding call
  and one reranker call for each independent group, with each query's original
  filters, candidate pool, min/max score normalization, thresholds and source
  deduplication. FAISS search remains single-query to retain its reference
  distance kernel. Failed batches fall back to serial retrieval. Implicit graph
  traversal remains serial because it can invoke the LLM.
- Bounded exact-schema grammar cache for the optional computation engine.
  Native validation and inference still run on every call; model outputs are
  never cached.
- Explicit device selection for both small retrieval models. CPU placement can
  reduce competition for unified GPU memory with Phi-3, but is an experiment,
  not an assumed accuracy-equivalent hardware change.
- Optional overlap of CPU indexing with the one initial Phi-3 question planner.
  Both complete before retrieval. Indexing never calls Phi-3, no second LLM is
  loaded, and exceptions join the worker before propagating. This overlap is
  disabled for LoCoMo's already-indexed query forks. It requires the separate
  `--pragmos-overlap-planner` flag and is NOT enabled by the recommended launcher:
  the bounded native probe did not demonstrate a speed benefit.

No prompt, source quota, reasoning stage, output budget, evidence threshold,
answer-promotion rule or model-state snapshot was removed. No concurrent calls
are made against the same Phi-3 context. The paused temporal/NLI experiment is
not enabled. The raw Phi-3 implementation is unchanged.

Batching and CPU/MPS changes can cause small floating-point differences. Unit
equivalence is not a guarantee of unchanged answers on all benchmark records.
Full-haystack QA parity and the 300-second target must be measured.

## Repeatable Development Runs

The launcher checks the frozen 20-question dataset and records current inference
hashes in the benchmark manifest. It uses new outputs; do not overwrite or resume
old V2/V3 results under changed code. A manifest mismatch requires a new run
version, not bypassing the guard.

```zsh
CODE="/Users/moitra/.codex/worktrees/lme-evidence-compiler/context"
/bin/zsh "$CODE/run_PRAGMOS_latency_20.zsh" optimized --check-only
caffeinate -i /bin/zsh "$CODE/run_PRAGMOS_latency_20.zsh" reference
caffeinate -i /bin/zsh "$CODE/run_PRAGMOS_latency_20.zsh" optimized
```

Both policies retain the existing diagnostic shadow computation. Its candidate
answers are NOT the returned answers or semantic judge accuracy. Do not claim
that a speedup came from disabling diagnostics. Keep both workloads identical;
compare actual answers, shadow candidate decisions and evidence separately.

Outputs:
`/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2/execution_latency_v1/runs/{reference,optimized}`

Use the average of trace `elapsed_seconds` for end-to-end question latency,
including ingestion. `execution_diagnostics` records index/planner wall time and
overlap if explicitly enabled. Do not add overlapping stage durations together. LoCoMo retains its
separate ingestion/fork accounting and serial prepared-query planner.

Acceptance requires no lost reference-correct semantic answers, unchanged
reasoning/evidence limits, and measured mean end-to-end time near 300 seconds.
Changed answers require semantic review or the existing judge. F1 is not that
gate. If those checks fail, retain reference execution and investigate before
running the larger suites.

## Verification

On 2026-11-14, the native CPU retrieval probe retained identical ranked source
IDs for seven queries over 24 documents. Serial retrieval took 0.156637 seconds;
batched retrieval took 0.111482 seconds (about 29% less retrieval-only time).
The largest score difference was 1.1921e-7. These tiny-document timings are not
a forecast of total question latency or evidence of benchmark accuracy.

Two real local Phi-3 planner calls over the same 32 invented turns verified
identical parsed/raw planner output, source records and indexed vectors with
index/planner overlap off and on. Serial preparation took 13.190 seconds versus
15.388 seconds overlapped. This small probe is not a controlled throughput study,
but it does not justify enabling overlap by default. No native library packages
or production import order were changed; the optional probe uses production's
Torch-before-FAISS initialization order.

The full unit suite has 1182 tests after adding the optional planner probe.
The final focused run had 417 passing tests and one skipped native-grammar
fixture; both separately enabled native execution probes passed.
The previously documented seven failures and one error remain (six older
whole-trace/prompt parity assertions, the paused NLI reason assertion, and the
old publication-freeze hash guard). These guards were not weakened. New policy
parity tests check actual prompts/calls, answers, selected evidence, aliases and
graphs across all six question families. A full benchmark/judge was not run.

Tests cover exact fake-model scores/order/provenance, thresholds, disabled
caches, retrieval ablations, batch failure fallback, cache invalidation, worker
cleanup, grammar validation, and full pipeline parity across all six families.
The optional native CPU probe loads only the installed retrieval models, not
Phi-3, and is not an end-to-end benchmark:

```zsh
cd /Users/moitra/.codex/worktrees/lme-evidence-compiler/context
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PRAGMOS_NATIVE_RETRIEVAL_TEST=1 \
  MPLCONFIGDIR=/tmp/pragmos-matplotlib PYTHONDONTWRITEBYTECODE=1 \
  /Users/moitra/surya/dev_project/context/.venv/bin/python -m unittest \
  test_PRAGMOS_execution_native
```
