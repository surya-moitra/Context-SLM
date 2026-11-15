#!/bin/zsh
set -eu
set -o pipefail

if (( $# < 1 || $# > 2 )) || [[ "$1" != reference && "$1" != optimized ]] || \
   [[ "${2:-}" != "" && "${2:-}" != --check-only ]]; then
  print -u2 "Usage: /bin/zsh run_PRAGMOS_latency_20.zsh reference|optimized [--check-only]"
  exit 2
fi

POLICY="$1"
CODE="${0:A:h}"
PROJECT="/Users/moitra/surya/dev_project/context"
PY="$PROJECT/.venv/bin/python"
MODEL="/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
PACK="$PROJECT/benchmark/longMemEval/synthetic_v2/semantic_revision_2"
DATA="$PACK/typed_computation_diagnostics_v1/pragmos_multisession_diagnostics_20_v1.json"
ROOT="$PACK/execution_latency_v1"
NAME="pragmos_diagnostics20_latency_${POLICY}_v1"
RUN="$ROOT/runs/$POLICY"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pragmos-latency-pycache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" -B - "$CODE" "$DATA" "$MODEL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

code, data, model = map(Path, sys.argv[1:])
for path in (code / "PRAGMOS_benchmark_LongMemEval.py", code / "PRAGMOS_execution.py", data, model):
    if not path.is_file():
        raise SystemExit("Missing required file: " + str(path))
if hashlib.sha256(data.read_bytes()).hexdigest() != "94df5216b1d62c28c1ec677e295802ca75670a2c271543cf7381c2380a2b4657":
    raise SystemExit("The frozen 20-question dataset changed; do not reuse this run version.")
records = json.loads(data.read_text())
if len(records) != 20 or len({r["question_id"] for r in records}) != 20:
    raise SystemExit("Expected twenty distinct development questions.")
print("Verified frozen diagnostic20 dataset and local files.")
print("This is a development timing run, not publication approval or semantic judge accuracy.")
print("Shadow computation remains enabled in BOTH policies; returned answers are the established path.")
print("Code/model/data/config hashes will be recorded by the benchmark manifest.")
PY

EXTRA_ARGS=(--pragmos-execution-policy "$POLICY")
if [[ "$POLICY" == optimized ]]; then
  EXTRA_ARGS+=(--pragmos-retrieval-device cpu)
fi
if [[ "${2:-}" == --check-only ]]; then
  "$PY" -B "$CODE/PRAGMOS_benchmark_LongMemEval.py" --help >/dev/null
  print "Preflight passed; no inference started. Policy: $POLICY"
  exit 0
fi

mkdir -p "$RUN"
"$PY" -B -u "$CODE/PRAGMOS_benchmark_LongMemEval.py" \
  --data-file "$DATA" --mode pragmos_context --question-type multi-session \
  --start-index 0 --limit 20 \
  --model-path "$MODEL" --n-ctx 2048 --n-threads 8 --n-gpu-layers 40 \
  --seed 42 --max-tokens 64 --temperature 0 --top-p 1 --repeat-penalty 1.1 \
  --pragmos-top-k 6 --pragmos-min-score 0.20 \
  --pragmos-graph-candidates 4 --pragmos-retrieval-pool 24 \
  --pragmos-multisession-graph-candidates 12 --pragmos-max-retrieval-sessions 12 \
  --pragmos-graph-depth 2 --pragmos-graph-evidence 4 \
  --pragmos-session-neighbor-radius 2 --pragmos-session-neighbors 4 \
  --pragmos-answer-context-tokens 2048 \
  --pragmos-chunk-words 160 --pragmos-chunk-overlap-words 32 \
  --pragmos-embedding-batch-size 64 --pragmos-evidence-compiler-mode validated \
  --pragmos-composable-mode shadow --pragmos-composable-protocol typed_v4 \
  --pragmos-temporal-engine contracts --pragmos-temporal-contract-mode off \
  "${EXTRA_ARGS[@]}" --flush-every 1 \
  --output-dir "$RUN" --run-name "$NAME" --resume \
  2>&1 | tee -a "$RUN/${NAME}_console.log"

print "Results: $RUN/${NAME}_summary.json"
print "Use elapsed_seconds for total latency; planner/index component timings overlap in optimized mode."
