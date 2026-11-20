#!/bin/zsh
set -eu
set -o pipefail

if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --check-only ]]; then
  print -u2 "Usage: /bin/zsh run_PRAGMOS_assistant_memory_projection_30.zsh [--check-only]"
  exit 2
fi
CODE="${0:A:h}"
PROJECT="/Users/moitra/surya/dev_project/context"
PY="$PROJECT/.venv/bin/python"
MODEL="/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
PACK="$PROJECT/benchmark/longMemEval/synthetic_v2/assistant_memory_30_v1"
DATA="$PACK/pragmos_assistant_memory_validation_30_v1.json"
RUN="$PACK/runs/source_projection_candidate_v1"
NAME="pragmos_assistant_memory_30_source_projection_candidate_v1"
ASSISTANCE="${PRAGMOS_ASSISTANCE_MODE:-off}"
if [[ "$ASSISTANCE" != off && "$ASSISTANCE" != guarded ]]; then
  print -u2 "PRAGMOS_ASSISTANCE_MODE must be off or guarded"
  exit 2
fi
if [[ "$ASSISTANCE" == guarded ]]; then
  RUN="$PACK/runs/guarded_advice_v1"
  NAME="pragmos_assistant_memory_30_guarded_advice_v1"
fi
print "Assistance mode: $ASSISTANCE"
print "Output directory: $RUN"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pragmos-assistant30-pycache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" -B - "$DATA" "$MODEL" <<'PY'
import hashlib
from pathlib import Path
import sys
data, model = map(Path, sys.argv[1:])
assert model.is_file(), "Model is missing"
assert hashlib.sha256(data.read_bytes()).hexdigest() == "b480590cfb1c45485f9e25310a774705fc701b1652e4235fad95aacbf905e3a6", "Assistant pack changed"
print("Frozen assistant30: same dataset and settings, new candidate output directory.")
PY
ARGS=(
  --data-file "$DATA" --mode pragmos_context --question-type single-session-assistant
  --start-index 0 --limit 30
  --model-path "$MODEL" --n-ctx 2048 --n-threads 8 --n-gpu-layers 40
  --seed 42 --max-tokens 64 --temperature 0 --top-p 1 --repeat-penalty 1.1
  --pragmos-top-k 6 --pragmos-min-score 0.20
  --pragmos-graph-candidates 4 --pragmos-retrieval-pool 24
  --pragmos-multisession-graph-candidates 12 --pragmos-max-retrieval-sessions 12
  --pragmos-graph-depth 2 --pragmos-graph-evidence 4
  --pragmos-session-neighbor-radius 2 --pragmos-session-neighbors 4
  --pragmos-answer-context-tokens 2048
  --pragmos-chunk-words 160 --pragmos-chunk-overlap-words 32
  --pragmos-embedding-batch-size 64 --pragmos-evidence-compiler-mode validated
  --pragmos-composable-mode candidate --pragmos-composable-protocol typed_v4
  --pragmos-temporal-engine contracts --pragmos-temporal-contract-mode off
  --pragmos-state-mode off --pragmos-execution-policy optimized
  --pragmos-retrieval-device cpu --pragmos-proof-first
  --pragmos-assistance-boundary off --pragmos-assistance-mode "$ASSISTANCE"
  --flush-every 1 --output-dir "$RUN" --run-name "$NAME" --resume
)
if [[ "${1:-}" == --check-only ]]; then
  "$PY" -B - "$CODE" "${ARGS[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from PRAGMOS_benchmark_LongMemEval import parse_args
from PRAGMOS_execution import validate_execution_arguments
validate_execution_arguments(parse_args())
print("Argument validation passed; no inference started.")
PY
  exit 0
fi
mkdir -p "$RUN"
"$PY" -B -u "$CODE/PRAGMOS_benchmark_LongMemEval.py" "${ARGS[@]}" \
  2>&1 | tee -a "$RUN/${NAME}_console.log"
print "Results: $RUN/${NAME}_summary.json"
if [[ "$ASSISTANCE" == guarded ]]; then
  print "Compare with source_projection_candidate_v1; actual answers, not shadow projections."
else
  print "Compare actual predictions with baseline_pre_cross_family_v1; no projected scores."
fi
