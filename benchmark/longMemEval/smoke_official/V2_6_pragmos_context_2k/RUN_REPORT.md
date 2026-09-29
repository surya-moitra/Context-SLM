# LongMemEval Official-Smoke Report - V2.6

Date: 2026-09-19

## Purpose

This is the final tightened-answer PRAGMOS smoke run on LongMemEval-S records
`5-9`, using the same question IDs, Phi-3 model, and 2K model context as the
raw Phi-3 haystack smoke.

The prediction JSONL has the official evaluator fields only:

```json
{"question_id": "...", "hypothesis": "..."}
```

The official GPT-4o judge was not run because `longmemeval_oracle.json` and
`OPENAI_API_KEY` are not present locally.

## Configuration

- Version: `V2.6 official-smoke`
- Mode: `pragmos_context`
- Dataset: `benchmark/longMemEval/longmemeval_s_cleaned.json` (`500` records)
- Slice: `--start-index 5 --limit 5`
- Model: `Phi-3-mini-4k-instruct-q4.gguf`
- Model context length: `2048`
- Final answer cap: `64` tokens
- PRAGMOS evidence cap for final answer: `1300` tokens
- Temperature: `0.0`
- Retrieval: dense + BM25 + graph expansion + session neighbors
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Graph candidates: `4`; graph depth: `2`; vector top-k: `6`
- Session neighbors: radius `2`, limit `4`

V2.6 adds a shared concise-answer policy, typed answer constraints, exact-value
preservation, query-matched old/new graph facts, and query-matched place/date
candidates from already extracted triples. These rules do not contain gold
answers or question IDs.

## Command

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

The command ran outside the sandbox so llama.cpp/Metal could initialize.

## Results

These string metrics are smoke diagnostics, not official LongMemEval scores.

- Processed: `5`
- Exact match: `0.400`
- Contains reference: `0.600`
- Token F1: `0.8267`
- Answer-session retrieval recall: `1.000`
- Total time: `444.33s`
- Average time: `88.87s/example`
- Average ingestion: `9.15s/example`
- Average retrieval: `58.61s/example`
- Average generation: `20.94s/example`

| Question | Gold | V2.6 hypothesis | Local F1 |
| --- | --- | --- | ---: |
| `c5e8278d` | Johnson | Johnson | `1.000` |
| `6ade9755` | Serenity Yoga | Serenity Yoga studio | `0.800` |
| `6f9b354f` | a lighter shade of gray | lighter shade of gray | `1.000` |
| `58ef2f1c` | February 14th | February (Valentine's Day) | `0.333` |
| `f8c5f88b` | the sports store downtown | downtown sports store | `1.000` |

All five hypotheses are semantically correct on manual inspection. The cheap
string metrics undercount valid equivalence: Valentine's Day is February 14th,
and `downtown sports store` reverses the gold word order without changing its
meaning. The official semantic judge is still required before making a score
claim.

## Comparison

Same five records and 2K model context:

| Run | Exact Match | Contains Ref | Token F1 | Answer-Session Recall | Avg Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw Phi-3 + haystack V1.0 | `0.000` | `0.200` | `0.0378` | n/a | `14.27s` |
| PRAGMOS V2.0 | `0.000` | `0.600` | `0.2298` | `1.000` | `97.48s` |
| PRAGMOS V2.6 | `0.400` | `0.600` | `0.8267` | `1.000` | `88.87s` |

V2.6 resolves the V2.0 temporal error (`Winters` instead of `Johnson`), removes
answer/evidence leakage, recovers the named yoga venue, preserves exact factual
spans, and substantially improves local F1 without changing retrieval recall.

## Files

- `pragmos_context_v2_6_official_smoke_predictions.jsonl`: official-compatible predictions.
- `pragmos_context_v2_6_official_smoke_trace.jsonl`: evidence, candidates, provenance, timings, and local metrics.
- `pragmos_context_v2_6_official_smoke_summary.json`: aggregate configuration and diagnostics.

