#!/bin/zsh
set -eu
set -o pipefail

if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --check-only ]]; then
  print -u2 "Usage: /bin/zsh run_PRAGMOS_single_user_35_odd.zsh [--check-only]"
  exit 2
fi
CODE="${0:A:h}"
PROJECT="/Users/moitra/surya/dev_project/context"
PY="$PROJECT/.venv/bin/python"
MODEL="/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
PACK="$PROJECT/benchmark/longMemEval/synthetic_v2/semantic_revision_2/single_session_user_35_odd_v1"
DATA="$PACK/pragmos_single_session_user_35_odd_v1.json"
RUN="$PACK/runs/factual_read_guarded_v3"
NAME="pragmos_user35_odd_factual_read_guarded_v3"
[[ -x "$PY" && -f "$MODEL" && -f "$DATA" && -f "$PACK/manifest.json" ]] || {
  print -u2 "Python, model, frozen pack or manifest is missing"
  exit 1
}
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pragmos-user35-odd-pycache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" -B - "$CODE" "$DATA" "$PACK/manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv.pop(1))
from PRAGMOS_factual_scope import resource_identity
if not resource_identity()['available']:
    raise RuntimeError('Guarded scope repair requires requirements-factual-read.txt and the local WordNet corpus')
payload = Path(sys.argv[1]).read_bytes()
manifest = json.loads(Path(sys.argv[2]).read_text())
expected = "b7d82bd57b00dd0619c9af2446b20c6d6a2ae3b2166f6b743531b5eeac05a42c"
if hashlib.sha256(payload).hexdigest() != expected or manifest["output_sha256"] != expected:
    raise ValueError("Frozen User35 dataset changed; refusing to mix workloads")
records = json.loads(payload)
if len(records) != 35 or any(r.get("question_type") != "single-session-user" for r in records):
    raise ValueError("Expected exactly 35 single-session-user records")
if [r["question_id"] for r in records] != manifest["validation"]["question_ids"]:
    raise ValueError("Pack order or question IDs differ from the manifest")
if manifest["selection_policy"]["positions_one_based"] != list(range(1, 71, 2)):
    raise ValueError("Expected original User70 positions 1, 3, ..., 69")
print("Frozen User35 verified: original User70 positions 1, 3, ..., 69.")
PY
print "Assistance mode: guarded; source-bound factual mode: guarded"
print "Model context: 2048; answer-context cap: 2048; output limit: 64"
print "Paused experimental temporal contract/checker remains off."
print "Output directory: $RUN"
ARGS=(
  --data-file "$DATA" --mode pragmos_context --question-type single-session-user
  --start-index 0 --limit 35
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
  --pragmos-assistance-boundary off --pragmos-assistance-mode guarded
  --pragmos-factual-mode guarded
  --flush-every 1 --output-dir "$RUN" --run-name "$NAME" --resume
)
if [[ "${1:-}" == --check-only ]]; then
  "$PY" -B - "$CODE" "${ARGS[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from PRAGMOS_benchmark_LongMemEval import parse_args
from PRAGMOS_execution import validate_execution_arguments
validate_execution_arguments(parse_args())
print("Arguments verified; no inference started.")
PY
  exit 0
fi
mkdir -p "$RUN"
"$PY" -B -u "$CODE/PRAGMOS_benchmark_LongMemEval.py" "${ARGS[@]}" 2>&1 | tee -a "$RUN/${NAME}_console.log"
print "Results: $RUN/${NAME}_summary.json"
print "Actual predictions; lexical metrics are not GPT-judge accuracy."
