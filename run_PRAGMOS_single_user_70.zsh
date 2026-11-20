#!/bin/zsh
set -eu
set -o pipefail

if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --check-only ]]; then
  print -u2 "Usage: PRAGMOS_ASSISTANCE_MODE=off|guarded /bin/zsh run_PRAGMOS_single_user_70.zsh [--check-only]"
  exit 2
fi
CODE="${0:A:h}"
PROJECT="/Users/moitra/surya/dev_project/context"
PY="$PROJECT/.venv/bin/python"
MODEL="/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
SYNTH="$PROJECT/benchmark/longMemEval/synthetic_v2/semantic_revision_2"
DATA="$SYNTH/pragmos_synthetic_longmemeval_500_v2_semantic_r2.json"
ASSISTANCE="${PRAGMOS_ASSISTANCE_MODE:-off}"
case "$ASSISTANCE" in
  guarded)
    RUN="$SYNTH/single_session_user_70_v1/runs/guarded_advice_v1"
    NAME="pragmos_user70_guarded_advice_v1"
    ;;
  off)
    RUN="$SYNTH/single_session_user_70_v1/runs/baseline_current_off_v1"
    NAME="pragmos_user70_current_off_v1"
    ;;
  *)
    print -u2 "PRAGMOS_ASSISTANCE_MODE must be off or guarded"
    exit 2
    ;;
esac
[[ -x "$PY" && -f "$MODEL" && -f "$DATA" ]] || { print -u2 "Python, model or dataset is missing"; exit 1; }
HASH="$(shasum -a 256 "$DATA")"
[[ "${HASH%% *}" == "080778e405d91f3b554ab5bb4b367e4eb2ccd4b751a5715559d7bd1a4f1bfaa9" ]] || {
  print -u2 "Frozen R2 dataset has changed; refusing to mix workloads"
  exit 1
}
COUNT="$(jq '[.[] | select(.question_type == "single-session-user")] | length' "$DATA")"
[[ "$COUNT" == 70 ]] || { print -u2 "Expected 70 single-session-user records, found $COUNT"; exit 1; }
print "Frozen R2 user70; assistance mode: $ASSISTANCE"
print "Output directory: $RUN"
print "Old User70 results use a different dataset/code; use matching current off/on runs for paired regression."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pragmos-user70-pycache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ARGS=(
  --data-file "$DATA" --mode pragmos_context --question-type single-session-user
  --start-index 0 --limit 70
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
print("Frozen User70 and arguments verified; no inference started.")
PY
  exit 0
fi
mkdir -p "$RUN"
"$PY" -B -u "$CODE/PRAGMOS_benchmark_LongMemEval.py" "${ARGS[@]}" 2>&1 | tee -a "$RUN/${NAME}_console.log"
print "Results: $RUN/${NAME}_summary.json"
print "Actual predictions; lexical metrics are not GPT-judge accuracy."
