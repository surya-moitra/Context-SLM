# LongMemEval Baseline Report - V0.0

Date: 2026-09-19

## Purpose

V0.0 is the question-only sanity baseline. Raw Phi-3 receives the LongMemEval
question and metadata only. It does not receive haystack sessions and does not
use PRAGMOS context.

This version is useful to confirm that the model is not answering from leaked
memory. It is not the fair comparison target for PRAGMOS.

## Configuration

- Version: `V0.0`
- Mode: `raw_phi3`
- Model: `Phi-3-mini-4k-instruct-q4.gguf`
- Context length: `2048`
- Max output tokens: `512`
- Temperature: `0.0`
- Examples: `5`
- Haystack sessions included: `false`

## Command

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

## Results

- Average exact match: `0.000`
- Average contains-reference: `0.000`
- Average token F1: `0.000`
- Total elapsed time: `16.69s`
- Average elapsed time: `3.34s/example`

## Interpretation

The zero score is expected. These first examples ask personal-memory questions
whose answers live in the haystack conversations. Since this baseline receives
no memory source, the correct behavior is usually to abstain rather than guess.

This run proves the no-context Phi-3 path is clean. The stronger baseline is
V1.0, where raw Phi-3 receives as much haystack text as can fit into the same
2K context window used by PRAGMOS.

## Files

- `phi3_question_only_v0_0_smoke_predictions.jsonl`: minimal predictions.
- `phi3_question_only_v0_0_smoke_trace.jsonl`: per-example trace with prompts,
  timing, answer, question type, and smoke metrics.
- `phi3_question_only_v0_0_smoke_summary.json`: aggregate settings and metrics.
