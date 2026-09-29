# LongMemEval Benchmark Report Index

Date: 2026-09-19

## Purpose

This folder contains the first versioned LongMemEval smoke benchmarks for raw
Phi-3 and PRAGMOS.

The goal is not to claim a final publishable score yet. The goal is to establish
clean, versioned baselines before comparing against PRAGMOS.

- `V0.0`: raw Phi-3 with only the question, no haystack memory.
- `V1.0`: raw Phi-3 with as much LongMemEval haystack as fits in the same 2K
  context window used by PRAGMOS.
- `V2.0`: Phi-3 with PRAGMOS indexing the complete haystack and injecting
  retrieved evidence into the same 2K context window.

## Dataset

- File: `benchmark/longMemEval/longmemeval_s_cleaned.json`
- Source: `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json`
- Size: 277,383,467 bytes
- Records: 500

Question type distribution:

- `knowledge-update`: 78
- `multi-session`: 133
- `single-session-assistant`: 56
- `single-session-preference`: 30
- `single-session-user`: 70
- `temporal-reasoning`: 133

## Context Alignment

PRAGMOS is configured with:

- `CONTEXT_LENGTH = 2048`
- `ANSWER_TOKEN_RESERVE = 512`
- `PROMPT_WRAPPER_TOKEN_RESERVE = 64`
- `CONTEXT_TOKEN_BUDGET = 1472`

The raw Phi-3 benchmark is also run with `--n-ctx 2048`, so V1.0 is the fair
raw-haystack baseline for the next PRAGMOS comparison.

## Versions

### V0.0 - Question Only

- Folder: `benchmark/longMemEval/V0_0_phi3_question_only`
- Mode: `raw_phi3`
- Haystack included: `false`
- Examples: `5`
- Average exact match: `0.000`
- Average token F1: `0.000`
- Average elapsed time: `3.34s/example`

Interpretation: expected zero. The model was not given the memory source.

### V1.0 - Raw Phi-3 + Haystack, 2K Window

- Folder: `benchmark/longMemEval/V1_0_phi3_haystack_2k`
- Mode: `raw_phi3_haystack`
- Haystack included: `true`
- Haystack packing: `recent-first`
- Examples: `5`
- Average exact match: `0.000`
- Average token F1: `0.000`
- Average elapsed time: `15.39s/example`

Interpretation: raw Phi-3 received haystack text, but only one recent session
fit per example under the 2K context limit. The relevant answer evidence was
not in that one included session, so naive haystack stuffing failed.

This is the baseline PRAGMOS should beat next.

### V2.0 - PRAGMOS + Complete Haystack, 2K Window

- Folder: `benchmark/longMemEval/V2_0_pragmos_context_2k`
- Mode: `pragmos_context`
- Complete haystack indexed: `true`
- Examples: `5`
- Average exact match: `0.000`
- Average contains-reference: `0.600`
- Average token F1: `0.2927`
- Answer-session retrieval recall: `1.000`
- Average elapsed time: `65.72s/example`

Interpretation: PRAGMOS beats the zero-score V1.0 raw-haystack baseline under
the same 2K model context limit. It retrieves the answer-bearing session for
all five examples and produces three answers containing the reference. The two
remaining failures are evidence selection/reasoning failures after successful
retrieval, not missing-haystack failures.

## Output Structure

Each version folder contains:

- `*_predictions.jsonl`: minimal evaluator-style predictions with
  `question_id` and `hypothesis`.
- `*_trace.jsonl`: detailed per-example trace with prompt estimates, timing,
  expected answer, local metrics, and haystack inclusion metadata where
  applicable.
- `*_summary.json`: aggregate settings, output paths, runtime, and smoke
  metrics.
- `RUN_REPORT.md`: human-readable version report.

## Commands

V0.0 question-only:

```bash
python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode raw_phi3 \
  --limit 5 \
  --output-dir benchmark/longMemEval/V0_0_phi3_question_only \
  --run-name phi3_question_only_v0_0_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048
```

V1.0 raw haystack:

```bash
python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode raw_phi3_haystack \
  --limit 5 \
  --output-dir benchmark/longMemEval/V1_0_phi3_haystack_2k \
  --run-name phi3_raw_haystack_v1_0_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048
```

V2.0 PRAGMOS context:

```bash
python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode pragmos_context \
  --limit 5 \
  --output-dir benchmark/longMemEval/V2_0_pragmos_context_2k \
  --run-name pragmos_context_v2_0_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048
```

## Notes And Caveats

The metrics are lightweight local string metrics for smoke testing. They are
useful for quick iteration, but the final benchmark should also use the official
LongMemEval evaluator or a task-appropriate scoring script.

These runs used only 5 examples, and the V2.0 sample was used during
development. The next benchmark should use a larger untouched sample before a
full 500-record run. Use the official LongMemEval evaluator for publication
claims.
