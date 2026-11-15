#!/bin/zsh
set -euo pipefail

CODE="${0:A:h}"
PROJECT="${PRAGMOS_PROJECT:-/Users/moitra/surya/dev_project/context}"
PY="${PRAGMOS_PYTHON:-$PROJECT/.venv/bin/python}"
MODEL="${PRAGMOS_MODEL:-/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf}"
PACK="$PROJECT/benchmark/longMemEval/synthetic_v2/knowledge_update_40_v1"
DATA="$CODE/benchmark/longMemEval/synthetic_v2/knowledge_update_40_v1/pragmos_knowledge_update_validation_40_v1.json"
TRACE_FIRST="$PACK/runs/baseline_production_v1/pragmos_ku40_baseline_production_2k_v1_trace.jsonl"
TRACE_REST="$PACK/runs/baseline_calendar_parsefix_remaining28_v2/pragmos_ku40_baseline_calendar_parsefix_remaining28_v2_trace.jsonl"
OUTPUT="${PRAGMOS_STATE_REPLAY_OUTPUT:-$PACK/runs/state_views_development/replay_full40_v1.json}"

[[ -x "$PY" ]] || { print -u2 "Python is not executable: $PY"; exit 1; }
for path in "$MODEL" "$DATA" "$TRACE_FIRST" "$TRACE_REST"; do
  [[ -f "$path" ]] || { print -u2 "Required file is missing: $path"; exit 1; }
done
[[ ! -e "$OUTPUT" && ! -e "${OUTPUT:r}.partial.json" ]] || {
  print -u2 "Report already exists. Set PRAGMOS_STATE_REPLAY_OUTPUT to a new version."
  exit 1
}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"

print "Saved-retrieval development replay, not a full benchmark or judge run."
exec "$PY" -B -u "$CODE/PRAGMOS_replay_state.py" \
  --data-file "$DATA" --trace "$TRACE_FIRST" --trace "$TRACE_REST" \
  --model-path "$MODEL" --n-ctx 2048 --n-threads 8 --n-gpu-layers 40 --seed 42 \
  --output "$OUTPUT" --progress
