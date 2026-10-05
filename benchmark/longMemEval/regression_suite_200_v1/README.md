# PRAGMOS LongMemEval Synthetic Regression Suite 200 V1

This frozen suite is a development regression gate for changes made while
generalizing PRAGMOS to other benchmarks such as LoCoMo. It is not an official
LongMemEval score and must not be reported as one.

## Coverage

| Question type | Questions |
| --- | ---: |
| Multi-session | 58 |
| Temporal reasoning | 50 |
| Knowledge update | 38 |
| Single-session user | 26 |
| Single-session assistant | 18 |
| Single-session preference | 10 |
| **Total** | **200** |

The suite retains all 30 synthetic abstention questions, all previously
observed zero-F1 multi-session cases, the historical knowledge-update failures,
and the exact cases used to validate temporal, state-history, ordinal-list, and
typed-slot fixes. Every positive synthetic subfamily remains represented.

The subset records are byte-for-value copies of their source records. The
manifest maps each subset index back to its original 0-based source index and
records why it was selected.

## Original source indices

The run command below uses the compact 200-record subset, so its runtime indices
are `0-199`. The corresponding 0-based indices in the original 500-record file
are:

- Single-session user: `6,8-11,21,23-24,27-30,32,35,38,53,55,57-58,60,64-69`
- Multi-session: `70-71,75,81,84-85,90-91,98,100-101,106,110-112,118,120-121,123,126-131,162,164,169-171,173-174,180-181,190-192,196-197,199-201,207-208,210-211,216,219-221,223,226-232`
- Preference: `136-137,140-141,143,147,154-155,158,160`
- Temporal: `235-236,238,242,248,251-253,256,261,263,266,269-270,280-282,284-286,292,298,300,302,306,309,311,315,319,324,326,330-332,337,340,342,345,347,349,351,355,357,359-365`
- Knowledge update: `368-370,372,375,397,403-415,419,422,424-427,429-441`
- Assistant memory: `444,447-450,452,455-456,458,469,477-478,481-482,487,495-497`

The manifest contains the complete uncompressed mapping and selection reason
for every record.

## Baseline run

Run this once before changing the PRAGMOS core for LoCoMo:

```bash
.venv/bin/python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/regression_suite_200_v1/pragmos_longmemeval_regression_200_v1.json \
  --mode pragmos_context \
  --start-index 0 \
  --limit 200 \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048 \
  --n-threads 8 \
  --n-gpu-layers 40 \
  --seed 42 \
  --max-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --repeat-penalty 1.1 \
  --pragmos-top-k 6 \
  --pragmos-min-score 0.20 \
  --pragmos-graph-candidates 4 \
  --pragmos-retrieval-pool 24 \
  --pragmos-multisession-graph-candidates 12 \
  --pragmos-max-retrieval-sessions 12 \
  --pragmos-graph-depth 2 \
  --pragmos-graph-evidence 4 \
  --pragmos-session-neighbor-radius 2 \
  --pragmos-session-neighbors 4 \
  --pragmos-answer-context-tokens 1300 \
  --pragmos-chunk-words 160 \
  --pragmos-chunk-overlap-words 32 \
  --pragmos-embedding-batch-size 64 \
  --flush-every 1 \
  --output-dir benchmark/longMemEval/regression_suite_200_v1/runs/baseline_pre_locomo \
  --run-name pragmos_lme_regression_200_baseline_pre_locomo \
  --resume
```

For each later candidate, keep every inference argument identical and change
only `--output-dir` and `--run-name`. Never resume a candidate run into the
baseline directory.

## Compare a candidate

After running a candidate into a separate directory, apply the strict gate:

```bash
.venv/bin/python compare_PRAGMOS_LongMemEval_regression.py \
  --baseline-trace benchmark/longMemEval/regression_suite_200_v1/runs/baseline_pre_locomo/pragmos_lme_regression_200_baseline_pre_locomo_trace.jsonl \
  --candidate-trace benchmark/longMemEval/regression_suite_200_v1/runs/CANDIDATE_DIRECTORY/CANDIDATE_RUN_NAME_trace.jsonl \
  --output benchmark/longMemEval/regression_suite_200_v1/runs/CANDIDATE_DIRECTORY/comparison_to_baseline.json
```

The default policy allows no individual question regression, no per-type mean
F1 regression, and no overall mean F1 regression. The command exits with status
1 when the gate fails and writes the individual improvements and regressions to
the comparison report. Use a nonzero tolerance only for a documented numerical
reason, not to hide a behavioral regression.

## Regenerate

```bash
.venv/bin/python build_PRAGMOS_LongMemEval_regression_suite.py
.venv/bin/python -m unittest \
  test_build_PRAGMOS_LongMemEval_regression_suite.py
```

## Freeze policy

Do not change V1 after its first baseline run. If the selection policy itself
must change, create V2 and retain V1 and all prior run outputs.
