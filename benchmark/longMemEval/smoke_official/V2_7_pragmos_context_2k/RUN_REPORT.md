# LongMemEval Official-Smoke Report - V2.7

Date: 2026-09-19

## Purpose

V2.7 is the frozen tightened-answer PRAGMOS smoke configuration. It uses
LongMemEval-S records `5-9`, identical question IDs, Phi-3 GGUF, and 2K model
context to the raw Phi-3 haystack baseline.

The prediction JSONL contains only the official evaluator fields
`question_id` and `hypothesis`. The official GPT-4o judge has not been run
because `longmemeval_oracle.json` and `OPENAI_API_KEY` are unavailable locally.

## Configuration

- Mode: `pragmos_context`
- Dataset: `benchmark/longMemEval/longmemeval_s_cleaned.json` (`500` records)
- Slice: `--start-index 5 --limit 5`
- Model: `Phi-3-mini-4k-instruct-q4.gguf`
- Model context: `2048` tokens
- Final answer cap: `64` tokens
- Final evidence cap: `1300` tokens
- Temperature: `0.0`
- Retrieval: dense + BM25 + graph expansion + session neighbors
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Graph candidates: `4`; graph depth: `2`; vector top-k: `6`
- Session neighbors: radius `2`, limit `4`

The final answer stage uses a shared concise-answer policy, typed answer
constraints, exact-value preservation, provenance-backed old/new facts, and
query-matched place/date candidates from already extracted triples. V2.7 also
rejects malformed place fragments and recognizes named businesses or venues
inside category phrases such as `retailer like X`. No question IDs or gold
answers are encoded in these rules.

## Command

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

The command ran outside the sandbox so llama.cpp/Metal could initialize.

## Results

These are cheap local string diagnostics, not official LongMemEval scores.

- Processed: `5`
- Exact match: `0.400`
- Contains reference: `0.600`
- Token F1: `0.8267`
- Answer-session retrieval recall: `1.000`
- Total time: `338.78s`
- Average time: `67.76s/example`
- Average ingestion: `6.05s/example`
- Average retrieval: `45.45s/example`
- Average generation: `16.20s/example`

| Question | Gold | V2.7 hypothesis | Local F1 |
| --- | --- | --- | ---: |
| `c5e8278d` | Johnson | Johnson | `1.000` |
| `6ade9755` | Serenity Yoga | Serenity Yoga studio | `0.800` |
| `6f9b354f` | a lighter shade of gray | lighter shade of gray | `1.000` |
| `58ef2f1c` | February 14th | February (Valentine's Day) | `0.333` |
| `f8c5f88b` | the sports store downtown | downtown sports store | `1.000` |

All five are semantically correct on manual inspection. The local string
metric undercounts Valentine's Day versus February 14th and reordered but
equivalent phrasing such as `downtown sports store`. The official semantic
judge remains necessary before making a benchmark score claim.

## Comparison

| Run | Exact Match | Contains Ref | Token F1 | Answer-Session Recall | Avg Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw Phi-3 + haystack V1.1 | `0.000` | `0.200` | `0.0478` | n/a | `13.83s` |
| PRAGMOS V2.0 | `0.000` | `0.600` | `0.2298` | `1.000` | `97.48s` |
| PRAGMOS V2.7 | `0.400` | `0.600` | `0.8267` | `1.000` | `67.76s` |

V1.1 is the primary smoke baseline because it shares V2.7's final answer policy
and 64-token output cap. V1.0 is retained only as a historical artifact.

## Independent Regression Check

Earlier records `0-4` were also used as a separate regression slice. The
final behavior returned exact answers for degree, commute duration, play, and
playlist. The one detected place error was isolated to a malformed graph
candidate; after the generic candidate-quality fix, a targeted rerun returned
the exact answer `Target` with source-turn provenance. This provides useful
smoke evidence against tuning only for records `5-9`, but it is still too small
to establish generalization. The next meaningful check should use an untouched
larger slice or the full 500 questions.

## Files

- `pragmos_context_v2_7_official_smoke_predictions.jsonl`: official-compatible predictions.
- `pragmos_context_v2_7_official_smoke_trace.jsonl`: retrieved evidence, structured candidates, provenance, timings, and local metrics.
- `pragmos_context_v2_7_official_smoke_summary.json`: aggregate configuration and diagnostics.
