# LongMemEval Official-Smoke Report - V1.0

Date: 2026-09-19

## Purpose

This run is the official-compatible smoke baseline for raw Phi-3 with haystack
stuffing. It uses a fresh five-question slice from LongMemEval-S, records
`5-9`, instead of the original first-five debug smoke.

The prediction file is written in the LongMemEval evaluator shape:

```json
{"question_id": "...", "hypothesis": "..."}
```

The actual GPT-4o judge score was not run in this environment because
`longmemeval_oracle.json` and `OPENAI_API_KEY` are not present locally.

## Configuration

- Version: `V1.0 official-smoke`
- Mode: `raw_phi3_haystack`
- Dataset: `benchmark/longMemEval/longmemeval_s_cleaned.json`
- Dataset size: `500`
- Slice: `--start-index 5 --limit 5`
- Question IDs: `c5e8278d`, `6ade9755`, `6f9b354f`, `58ef2f1c`, `f8c5f88b`
- Model: `Phi-3-mini-4k-instruct-q4.gguf`
- Context length: `2048`
- Max output tokens: `512`
- Temperature: `0.0`
- Haystack order: `recent-first`

## Command

```bash
.venv/bin/python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode raw_phi3_haystack \
  --start-index 5 \
  --limit 5 \
  --output-dir benchmark/longMemEval/smoke_official/V1_0_phi3_haystack_2k \
  --run-name phi3_raw_haystack_v1_0_official_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048 \
  --flush-every 1
```

The first sandboxed attempt failed with `Failed to create llama_context`, so the
same command was rerun outside the sandbox to allow llama.cpp/Metal to
initialize the local model.

## Local Smoke Metrics

These are cheap local string metrics only. They are useful for catching broken
runs, but they are not the official LongMemEval score.

- Processed examples: `5`
- Average exact match: `0.000`
- Average contains-reference: `0.200`
- Average token F1: `0.0378`
- Total elapsed time: `71.36s`
- Average elapsed time: `14.27s/example`

Per-example local output:

```text
[1] c5e8278d f1=0.000 em=0 time=13.13s
[2] 6ade9755 f1=0.118 em=0 time=15.62s
[3] 6f9b354f f1=0.000 em=0 time=12.12s
[4] 58ef2f1c f1=0.071 em=0 time=15.41s
[5] f8c5f88b f1=0.000 em=0 time=14.35s
```

## Observations

With a 2K context window, raw Phi-3 could include only one recent haystack
session per question:

- `c5e8278d`: 44 sessions total, 1 included, answer `Johnson`, missed.
- `6ade9755`: 51 sessions total, 1 included, answer `Serenity Yoga`, found.
- `6f9b354f`: 50 sessions total, 1 included, answer `a lighter shade of gray`, missed.
- `58ef2f1c`: 50 sessions total, 1 included, answer `February 14th`, partial answer `February`.
- `f8c5f88b`: 46 sessions total, 1 included, answer `the sports store downtown`, missed.

This is the behavior we want from the raw baseline before running PRAGMOS: the
baseline is constrained by the same small context window and cannot search the
full haystack. PRAGMOS should use the same model/window, but ingest all
haystack turns and inject only selected evidence.

## Official Evaluation Readiness

Generated prediction file:

- `phi3_raw_haystack_v1_0_official_smoke_predictions.jsonl`

Expected official evaluator command after downloading the official
LongMemEval evaluator and oracle file:

```bash
python /path/to/LongMemEval/src/evaluation/evaluate_qa.py \
  gpt-4o \
  benchmark/longMemEval/smoke_official/V1_0_phi3_haystack_2k/phi3_raw_haystack_v1_0_official_smoke_predictions.jsonl \
  benchmark/longMemEval/longmemeval_oracle.json
```

## Files

- `phi3_raw_haystack_v1_0_official_smoke_predictions.jsonl`: official-compatible predictions.
- `phi3_raw_haystack_v1_0_official_smoke_trace.jsonl`: per-example trace, answers, included haystack sessions, prompt token estimates, and local smoke metrics.
- `phi3_raw_haystack_v1_0_official_smoke_summary.json`: aggregate config, metrics, and official-evaluation readiness metadata.
