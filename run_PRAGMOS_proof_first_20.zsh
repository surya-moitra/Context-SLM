#!/bin/zsh
set -eu
set -o pipefail

if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != --check-only ]]; then
  print -u2 "Usage: /bin/zsh run_PRAGMOS_proof_first_20.zsh [--check-only]"
  exit 2
fi
CODE="${0:A:h}"
PROJECT="/Users/moitra/surya/dev_project/context"
PY="$PROJECT/.venv/bin/python"
MODEL="/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
PACK="$PROJECT/benchmark/longMemEval/synthetic_v2/semantic_revision_2"
DATA="$PACK/typed_computation_diagnostics_v1/pragmos_multisession_diagnostics_20_v1.json"
RUN="$PACK/execution_latency_v2/runs/proof_first"
NAME="pragmos_diagnostics20_proof_first_candidate_v2"
ASSISTANCE="${PRAGMOS_ASSISTANCE_MODE:-off}"
if [[ "$ASSISTANCE" != off && "$ASSISTANCE" != guarded ]]; then
  print -u2 "PRAGMOS_ASSISTANCE_MODE must be off or guarded"
  exit 2
fi
if [[ "$ASSISTANCE" == guarded ]]; then
  RUN="$PACK/execution_latency_v2/runs/guarded_advice_v1"
  NAME="pragmos_diagnostics20_guarded_advice_v1"
fi
print "Assistance mode: $ASSISTANCE"
print "Output directory: $RUN"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pragmos-proof-first-pycache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" -B - "$CODE" "$DATA" "$MODEL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

code, data, model = map(Path, sys.argv[1:])
for path in (code / "PRAGMOS_benchmark_LongMemEval.py", code / "PRAGMOS_proof_execution.py", data, model):
    if not path.is_file():
        raise SystemExit("Missing required file: " + str(path))
if hashlib.sha256(data.read_bytes()).hexdigest() != "94df5216b1d62c28c1ec677e295802ca75670a2c271543cf7381c2380a2b4657":
    raise SystemExit("Frozen diagnostic20 dataset changed; do not reuse this run version.")
records = json.loads(data.read_text())
assert len(records) == len({r["question_id"] for r in records}) == 20
print("Verified the same frozen 20 questions. Candidate answers will be RETURNED, not shadowed.")
print("Positive proof first; legacy fallback otherwise. Graph materialization is unchanged.")
print("Development validation only: correctness and sub-300-second latency are not guaranteed.")
PY

ARGS=(
  --data-file "$DATA" --mode pragmos_context --question-type multi-session
  --start-index 0 --limit 20
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
print("Full argument validation passed; no inference started.")
PY
  exit 0
fi
mkdir -p "$RUN"
"$PY" -B -u "$CODE/PRAGMOS_benchmark_LongMemEval.py" "${ARGS[@]}" \
  2>&1 | tee -a "$RUN/${NAME}_console.log"
print "Results: $RUN/${NAME}_summary.json"
print "These predictions are actual candidate-enabled answers, not shadow projections."
