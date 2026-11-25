#!/bin/zsh
set -eu
set -o pipefail

if (( $# < 1 || $# > 2 )) || \
   [[ "$1" != multisession20 && "$1" != user35even && "$1" != assistant30 && \
      "$1" != preferences30 && "$1" != knowledge40 && "$1" != temporal50 && "$1" != mixed200 ]] || \
   [[ "${2:-}" != "" && "${2:-}" != --check-only ]]; then
  print -u2 "Usage: /bin/zsh run_PRAGMOS_factual_guard_regressions.zsh multisession20|user35even|assistant30|preferences30|knowledge40|temporal50|mixed200 [--check-only]"
  exit 2
fi

SUITE="$1"
CODE="${0:A:h}"
PROJECT="/Users/moitra/surya/dev_project/context"
PY="$PROJECT/.venv/bin/python"
MODEL="/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
SYNTH="$PROJECT/benchmark/longMemEval/synthetic_v2/semantic_revision_2"
case "$SUITE" in
  multisession20)
    DATA="$SYNTH/typed_computation_diagnostics_v1/pragmos_multisession_diagnostics_20_v1.json"
    EXPECTED="94df5216b1d62c28c1ec677e295802ca75670a2c271543cf7381c2380a2b4657"
    COUNT=20
    QUESTION_TYPE=multi-session
    RUN="$SYNTH/execution_latency_v2/runs/factual_read_guarded_v3"
    NAME="pragmos_diagnostics20_factual_read_guarded_v3"
    ;;
  user35even)
    DATA="$SYNTH/single_session_user_35_even_v1/pragmos_single_session_user_35_even_v1.json"
    EXPECTED="0227ade1e7ede35c527583ef52d3aa922977ab834d3b08a76e07bef34c850931"
    COUNT=35
    QUESTION_TYPE=single-session-user
    RUN="$SYNTH/single_session_user_35_even_v1/runs/factual_read_guarded_v3"
    NAME="pragmos_user35_even_factual_read_guarded_v3"
    ;;
  assistant30)
    DATA="$PROJECT/benchmark/longMemEval/synthetic_v2/assistant_memory_30_v1/pragmos_assistant_memory_validation_30_v1.json"
    EXPECTED="b480590cfb1c45485f9e25310a774705fc701b1652e4235fad95aacbf905e3a6"
    COUNT=30
    QUESTION_TYPE=single-session-assistant
    RUN="$PROJECT/benchmark/longMemEval/synthetic_v2/assistant_memory_30_v1/runs/factual_read_guarded_v3"
    NAME="pragmos_assistant30_factual_read_guarded_v3"
    ;;
  preferences30)
    DATA="$SYNTH/pragmos_synthetic_longmemeval_500_v2_semantic_r2.json"
    EXPECTED="080778e405d91f3b554ab5bb4b367e4eb2ccd4b751a5715559d7bd1a4f1bfaa9"
    COUNT=30
    QUESTION_TYPE=single-session-preference
    RUN="$SYNTH/preferences_30_v1/runs/factual_read_guarded_v3"
    NAME="pragmos_preferences30_factual_read_guarded_v3"
    ;;
  knowledge40)
    DATA="$CODE/benchmark/longMemEval/synthetic_v2/knowledge_update_40_v1/pragmos_knowledge_update_validation_40_v1.json"
    EXPECTED="65431f373ab32a77db19897639499e7fe042ea152ca46b8af71daac60adf4aa1"
    COUNT=40
    QUESTION_TYPE=knowledge-update
    RUN="$PROJECT/benchmark/longMemEval/synthetic_v2/knowledge_update_40_v1/runs/factual_read_guarded_v3"
    NAME="pragmos_knowledge40_factual_read_guarded_v3"
    ;;
  temporal50)
    DATA="$PROJECT/benchmark/longMemEval/synthetic_v2/temporal_revision_1/pragmos_lme_v2_temporal_validation_50_r1.json"
    EXPECTED="7df7f054f3bd482de405bc2095f876f42d65bc6e149017b39a706646ec46160d"
    COUNT=50
    QUESTION_TYPE=temporal-reasoning
    RUN="$PROJECT/benchmark/longMemEval/synthetic_v2/temporal_revision_1/runs/factual_read_guarded_v3"
    NAME="pragmos_temporal50_factual_read_guarded_v3"
    ;;
  mixed200)
    DATA="$PROJECT/benchmark/longMemEval/regression_suite_200_v1/pragmos_longmemeval_regression_200_v1.json"
    EXPECTED="317695f45a10bd4f94ffe6cd7a43c869c4171b375f6765ac912b8032feaa3f64"
    COUNT=200
    QUESTION_TYPE=""
    RUN="$PROJECT/benchmark/longMemEval/regression_suite_200_v1/runs/factual_read_guarded_v3"
    NAME="pragmos_mixed200_factual_read_guarded_v3"
    ;;
esac
[[ -x "$PY" && -f "$MODEL" && -f "$DATA" ]] || {
  print -u2 "Python, model or frozen dataset is missing"
  exit 1
}
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pragmos-matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pragmos-factual-regression-pycache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" -B - "$CODE" "$DATA" "$EXPECTED" "$COUNT" "$QUESTION_TYPE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv.pop(1))
from PRAGMOS_factual_scope import resource_identity
if not resource_identity()['available']:
    raise RuntimeError('Install the pinned local factual-scope dependencies and WordNet before running')
path, expected, count, question_type = sys.argv[1:]
payload = Path(path).read_bytes()
if hashlib.sha256(payload).hexdigest() != expected:
    raise ValueError('Frozen regression dataset changed; refusing to mix workloads')
records = json.loads(payload)
if question_type:
    records = [record for record in records if record.get('question_type') == question_type]
if (len(records) != int(count) or len({r['question_id'] for r in records}) != int(count)
        or (question_type and any(r.get('question_type') != question_type for r in records))):
    raise ValueError('Regression count, IDs or question types do not match the frozen suite')
print(f'Frozen {question_type or "mixed-type"} suite verified: {count} unchanged questions.')
PY
print "Assistance mode: guarded; source-bound factual mode: guarded"
print "Actual candidate-enabled answers, not shadow projections."
print "Model context: 2048; answer-context cap: 2048; output limit: 64"
print "Paused experimental temporal contract/checker remains off."
print "Experimental state resolver remains off; this is a full haystack run, not saved-evidence replay."
print "Output directory: $RUN"
ARGS=(
  --data-file "$DATA" --mode pragmos_context
  --start-index 0 --limit "$COUNT"
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
if [[ -n "$QUESTION_TYPE" ]]; then
  ARGS+=(--question-type "$QUESTION_TYPE")
fi
if [[ "${2:-}" == --check-only ]]; then
  "$PY" -B - "$CODE" "${ARGS[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from PRAGMOS_benchmark_LongMemEval import parse_args
from PRAGMOS_execution import validate_execution_arguments
args = parse_args()
validate_execution_arguments(args)
assert args.pragmos_assistance_mode == args.pragmos_factual_mode == 'guarded'
assert args.pragmos_composable_mode == 'candidate'
assert args.pragmos_temporal_contract_mode == 'off'
assert args.pragmos_state_mode == 'off'
assert args.pragmos_proof_first and args.pragmos_composable_protocol == 'typed_v4'
print('Arguments verified; no inference started.')
PY
  exit 0
fi
mkdir -p "$RUN"
"$PY" -B -u "$CODE/PRAGMOS_benchmark_LongMemEval.py" "${ARGS[@]}" 2>&1 | tee -a "$RUN/${NAME}_console.log"
print "Results: $RUN/${NAME}_summary.json"
print "Actual predictions; lexical metrics are not GPT-judge accuracy."
