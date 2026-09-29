# LongMemEval Official-Smoke Report Index

Date: 2026-09-19

This folder contains smoke runs prepared in the LongMemEval official prediction
format. These runs are intended to validate the benchmark pipeline before the
full 500-question LongMemEval-S benchmark.

## Runs

- `V1_0_phi3_haystack_2k`: raw Phi-3 with as much haystack as fits inside a
  2K context window, using records `5-9`.
- `V1_1_phi3_haystack_2k_answer_policy`: fair raw Phi-3 comparison using the
  final shared answer policy and 64-token output cap on records `5-9`.
- `V2_0_pragmos_context_2k`: PRAGMOS + Phi-3 with full haystack ingestion,
  retrieval, graph evidence, and a bounded 2K answer prompt, using the same
  records `5-9`.
- `V2_6_pragmos_context_2k`: tightened PRAGMOS answer generation with a shared
  answer policy, 64-token output cap, 1300-token evidence cap, structured
  temporal facts, and typed place/date candidates on records `5-9`.
- `V2_7_pragmos_context_2k`: frozen final-smoke configuration; adds generic
  place-candidate quality filtering and named venue/business extraction.

## Current Status

The V1.0 raw Phi-3 haystack smoke completed successfully and produced an
official-compatible prediction JSONL with `question_id` and `hypothesis`.

V1.1 reran raw Phi-3 with V2.7's final shared answer settings. Use V1.1 as the
primary apples-to-apples smoke baseline; V1.0 is historical.

The V2.0 PRAGMOS smoke completed successfully and produced an
official-compatible prediction JSONL for the same five questions.

The V2.6 PRAGMOS smoke also completed successfully. All five hypotheses are
semantically correct on manual inspection, answer-session retrieval recall is
`1.000`, and local token F1 increased from `0.2298` to `0.8267`.

V2.7 reproduces the same semantic answer quality with the final code and is the
configuration to carry into the next untouched/full benchmark run.

The GPT-4o judge score has not been run yet because the local folder does not
currently contain `longmemeval_oracle.json`, and `OPENAI_API_KEY` is not set in
this shell environment.

## Same-Slice Local Comparison

These are cheap local smoke metrics only, not official LongMemEval judge
scores.

| Run | Contains Ref | Token F1 | Answer-Session Retrieval | Avg Time |
| --- | ---: | ---: | ---: | ---: |
| Raw Phi-3 + haystack V1.0 (historical) | `0.200` | `0.0378` | n/a | `14.27s` |
| Raw Phi-3 + haystack V1.1 (fair baseline) | `0.200` | `0.0478` | n/a | `13.83s` |
| PRAGMOS + haystack V2.0 | `0.600` | `0.2298` | `1.000` | `97.48s` |
| PRAGMOS + haystack V2.6 | `0.600` | `0.8267` | `1.000` | `88.87s` |
| PRAGMOS + haystack V2.7 | `0.600` | `0.8267` | `1.000` | `67.76s` |

## Reproduction Commands

Raw Phi-3 + haystack:

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

Fair raw Phi-3 V1.1:

```bash
.venv/bin/python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode raw_phi3_haystack \
  --start-index 5 \
  --limit 5 \
  --output-dir benchmark/longMemEval/smoke_official/V1_1_phi3_haystack_2k_answer_policy \
  --run-name phi3_raw_haystack_v1_1_official_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048 \
  --flush-every 1
```

PRAGMOS + haystack:


```bash
.venv/bin/python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode pragmos_context \
  --start-index 5 \
  --limit 5 \
  --output-dir benchmark/longMemEval/smoke_official/V2_0_pragmos_context_2k \
  --run-name pragmos_context_v2_0_official_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048 \
  --flush-every 1
```

Final tightened PRAGMOS V2.6:

```bash
.venv/bin/python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode pragmos_context \
  --start-index 5 \
  --limit 5 \
  --output-dir benchmark/longMemEval/smoke_official/V2_6_pragmos_context_2k \
  --run-name pragmos_context_v2_6_official_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048 \
  --flush-every 1
```

Frozen final-smoke PRAGMOS V2.7:

```bash
.venv/bin/python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode pragmos_context \
  --start-index 5 \
  --limit 5 \
  --output-dir benchmark/longMemEval/smoke_official/V2_7_pragmos_context_2k \
  --run-name pragmos_context_v2_7_official_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048 \
  --flush-every 1
```
