# LongMemEval Raw Phi-3 Official-Smoke Report - V1.1

Date: 2026-09-19

## Purpose

V1.1 is the fair raw Phi-3 haystack baseline for PRAGMOS V2.7. Both runs use
the same five LongMemEval-S records, Phi-3 GGUF, 2K model context, deterministic
decoding, shared concise-answer policy, and 64-token output cap.

Raw Phi-3 receives as many recent haystack sessions as fit directly in the 2K
prompt. It does not import or use the PRAGMOS context layer.

## Command

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

## Results

- Processed: `5`
- Exact match: `0.000`
- Contains reference: `0.200`
- Token F1: `0.0478`
- Total time: `69.15s`
- Average time: `13.83s/example`
- Haystack sessions included: `1` per question

These are local string diagnostics, not official LongMemEval scores. Raw Phi-3
found `Serenity Yoga` and the month `February`, but missed the other answers and
often produced refusal text. The official semantic judge has not been run.

## Files

- `phi3_raw_haystack_v1_1_official_smoke_predictions.jsonl`: official-compatible predictions.
- `phi3_raw_haystack_v1_1_official_smoke_trace.jsonl`: prompt-fit diagnostics, timings, and local metrics.
- `phi3_raw_haystack_v1_1_official_smoke_summary.json`: aggregate configuration and diagnostics.

