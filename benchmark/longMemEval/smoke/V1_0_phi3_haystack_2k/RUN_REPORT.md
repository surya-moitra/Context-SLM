# LongMemEval Baseline Report - V1.0

Date: 2026-09-19

## Purpose

V1.0 is the raw Phi-3 plus haystack-stuffing baseline. Raw Phi-3 receives the
question plus as many LongMemEval haystack sessions as fit inside the same
`2048` context window configured in PRAGMOS.

This is the apples-to-apples raw baseline for the next PRAGMOS run. PRAGMOS
should use the same model and context window, but replace naive haystack
stuffing with graph memory ingestion, retrieval, and evidence formatting.

## Configuration

- Version: `V1.0`
- Mode: `raw_phi3_haystack`
- Model: `Phi-3-mini-4k-instruct-q4.gguf`
- Context length: `2048`
- Max output tokens: `512`
- Temperature: `0.0`
- Haystack order: `recent-first`
- Examples: `5`
- Haystack sessions included: `true`

## Command

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

In this Codex app session, the command was run outside the sandbox so llama.cpp
could initialize the Mac Metal backend.

## Results

- Average exact match: `0.000`
- Average contains-reference: `0.000`
- Average token F1: `0.000`
- Total elapsed time: `76.96s`
- Average elapsed time: `15.39s/example`

Per-example output:

```text
[1] e47becba f1=0.000 em=0 time=17.83s
[2] 118b2229 f1=0.000 em=0 time=14.14s
[3] 51a45a95 f1=0.000 em=0 time=14.90s
[4] 58bf7951 f1=0.000 em=0 time=14.94s
[5] 1e043500 f1=0.000 em=0 time=14.80s
```

## Interpretation

This is the important baseline. Raw Phi-3 was given haystack text, but the 2K
window only allowed one recent session to fit for each of the first five
examples. Those records each had many haystack sessions:

- `e47becba`: 53 total, 1 included
- `118b2229`: 45 total, 1 included
- `51a45a95`: 50 total, 1 included
- `58bf7951`: 57 total, 1 included
- `1e043500`: 50 total, 1 included

The model mostly answered that the requested fact was not present in the
provided history. That is a useful failure: naive recency-first stuffing cannot
find the right memory when the relevant evidence is outside the small context
window.

For PRAGMOS, the expected win is not from giving the model more context. The
expected win is from choosing better context: ingest all haystack turns into the
graph, retrieve the evidence-bearing memories, and inject only those memories
within the same 2K budget.

## Files

- `phi3_raw_haystack_v1_0_smoke_predictions.jsonl`: minimal predictions.
- `phi3_raw_haystack_v1_0_smoke_trace.jsonl`: per-example trace including
  included haystack session IDs, included session indices, estimated prompt
  tokens, timing, answers, and smoke metrics.
- `phi3_raw_haystack_v1_0_smoke_summary.json`: aggregate settings and metrics.
