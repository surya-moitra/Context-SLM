# LongMemEval Official-Smoke Report - V2.0

Date: 2026-09-19

## Purpose

This run is the official-compatible PRAGMOS smoke benchmark on the same
five-question LongMemEval-S slice used by the raw Phi-3 haystack smoke:
records `5-9`.

The prediction file is written in the LongMemEval evaluator shape:

```json
{"question_id": "...", "hypothesis": "..."}
```

The actual GPT-4o judge score was not run in this environment because
`longmemeval_oracle.json` and `OPENAI_API_KEY` are not present locally.

## Configuration

- Version: `V2.0 official-smoke`
- Mode: `pragmos_context`
- Dataset: `benchmark/longMemEval/longmemeval_s_cleaned.json`
- Dataset size: `500`
- Slice: `--start-index 5 --limit 5`
- Question IDs: `c5e8278d`, `6ade9755`, `6f9b354f`, `58ef2f1c`, `f8c5f88b`
- Model: `Phi-3-mini-4k-instruct-q4.gguf`
- Context length: `2048`
- Max output tokens: `512`
- Temperature: `0.0`
- Retrieval: dense + BM25 + graph expansion + cross-encoder reranker
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Graph candidates: `4`
- Graph depth: `2`
- Vector top-k: `6`
- Session neighbors: radius `2`, limit `4`

## Command

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

The command was run outside the sandbox so llama.cpp/Metal and local embedding
components could initialize correctly.

## Local Smoke Metrics

These are cheap local string metrics only. They are useful for catching broken
runs, but they are not the official LongMemEval score.

- Processed examples: `5`
- Average exact match: `0.000`
- Average contains-reference: `0.600`
- Average token F1: `0.2298`
- Answer-session retrieval recall: `1.000`
- Total elapsed time: `487.38s`
- Average elapsed time: `97.48s/example`
- Average ingestion time: `10.10s/example`
- Average retrieval time: `66.93s/example`
- Average generation time: `20.27s/example`

Per-example local output:

```text
[1] c5e8278d f1=0.035 em=0 time=82.65s
[2] 6ade9755 f1=0.070 em=0 time=108.69s
[3] 6f9b354f f1=0.615 em=0 time=98.69s
[4] 58ef2f1c f1=0.143 em=0 time=107.85s
[5] f8c5f88b f1=0.286 em=0 time=89.10s
```

## Comparison With Raw Phi-3 + Haystack

Same five records, same Phi-3 GGUF, same 2K context window:

| Run | Contains Ref | Token F1 | Answer-Session Retrieval | Avg Time |
| --- | ---: | ---: | ---: | ---: |
| Raw Phi-3 + haystack | `0.200` | `0.0378` | n/a | `14.27s` |
| PRAGMOS + haystack | `0.600` | `0.2298` | `1.000` | `97.48s` |

The smoke shows the intended PRAGMOS effect: it can ingest the full haystack,
retrieve answer-bearing evidence, and improve over naive one-session stuffing
inside the same 2K model context.

## Per-Question Observations

- `c5e8278d`: retrieved the correct session and included the source quote with
  old name `Johnson`, but Phi-3 answered the current last name `Winters`. This
  is a temporal answer-selection failure, not a retrieval failure.
- `6ade9755`: retrieved the correct session and included `Serenity Yoga`, but
  the answer was verbose and partially self-contradictory.
- `6f9b354f`: strong answer: `a lighter shade of gray`.
- `58ef2f1c`: retrieved the correct session but answered only `February`,
  missing `14th`.
- `f8c5f88b`: retrieved enough evidence to answer, but produced a typo-like
  variant: `sports store downt Town`.

## Interpretation

This is a successful pipeline smoke, but not yet a publication-quality final
configuration.

The retrieval layer is doing the most important thing correctly on this slice:
it hit the answer-bearing session for all five questions. That is the core
PRAGMOS claim starting to show up.

The answer generator still needs tightening before the full 500-question run:

- enforce answer-only output more strongly;
- improve temporal slot selection for questions like `before`, `previous`,
  `earlier`, and `changed from`;
- reduce evidence leakage into the final `hypothesis`;
- consider a smaller `max_tokens` for benchmark answering, because the current
  `512` token budget invites verbose outputs.

## Official Evaluation Readiness

Generated prediction file:

- `pragmos_context_v2_0_official_smoke_predictions.jsonl`

Expected official evaluator command after downloading the official
LongMemEval evaluator and oracle file:

```bash
python /path/to/LongMemEval/src/evaluation/evaluate_qa.py \
  gpt-4o \
  benchmark/longMemEval/smoke_official/V2_0_pragmos_context_2k/pragmos_context_v2_0_official_smoke_predictions.jsonl \
  benchmark/longMemEval/longmemeval_oracle.json
```

## Files

- `pragmos_context_v2_0_official_smoke_predictions.jsonl`: official-compatible predictions.
- `pragmos_context_v2_0_official_smoke_trace.jsonl`: per-example trace, retrieved memories, graph evidence, provenance, answer-session recall, timing, and local smoke metrics.
- `pragmos_context_v2_0_official_smoke_summary.json`: aggregate config, metrics, and official-evaluation readiness metadata.
