# PRAGMOS Measurement Harness

This sidecar harness runs an existing benchmark command unchanged. It does not
import or modify PRAGMOS. Publication runs are guarded against the frozen core
tag `pragmos-lme-locomo-official-candidate-v1` by comparing SHA-256 hashes for:

- `PRAGMOS_context_layer_org.py`
- `PRAGMOS_benchmark_LongMemEval.py`
- `Phi3_raw_baseline.py`

## Install

From the activated project virtual environment:

```bash
python -m pip install -r requirements-measurement.txt
```

## Run

Wrap the normal benchmark command after `--`:

```bash
caffeinate -i python PRAGMOS_measurement_harness.py \
  --run-name pragmos_publication_run \
  --output-dir benchmark/measurements/pragmos_publication_run \
  --benchmark-summary path/to/the_benchmark_summary.json \
  --sample-interval 0.5 \
  --local-only \
  -- \
  python PRAGMOS_benchmark_LongMemEval.py [the normal benchmark arguments]
```

The benchmark still prints progress normally. Its own `--resume` option remains
usable. For a resumed invocation, use a new measurement run name; CPU time per
question uses the benchmark summary's `newly_processed` count.

The harness writes:

- `*_measurement_manifest.json`: command, Git/core identities, hardware, local
  execution controls, and artifact paths.
- `*_resource_samples.jsonl`: raw timestamped RAM samples.
- `*_resource_summary.json`: the three headline resource metrics.

The resource metrics are deliberately limited to:

1. Peak sampled process-tree RSS in GiB.
2. Mean CPU utilization as a percentage of total machine CPU capacity.
3. CPU seconds per question processed during that invocation.

If macOS denies descendant enumeration, the result is marked as an incomplete
process-tree measurement and the RAM value is explicitly reported as a lower
bound.

## Compact Report

After the benchmark and optional official judge have completed:

```bash
python PRAGMOS_measurement_report.py \
  --benchmark-summary path/to/the_benchmark_summary.json \
  --resource-summary path/to/the_resource_summary.json \
  --judge-summary path/to/the_official_judge_summary.json \
  --trace-metrics path/to/trace_measurements.json \
  --output-json path/to/measurement_report.json \
  --output-markdown path/to/measurement_report.md
```

The compact report contains the existing F1, judge accuracy, retrieval, and
latency results plus the resource, context-efficiency, and memory/graph quality
metrics. It rejects mismatched benchmark and resource invocation counts.

## Trace Metrics

New LoCoMo runs write a versioned `measurement_trace` object into every trace
row and aggregate it into the normal benchmark summary. This changes only trace
serialization and post-inference accounting. Retrieval and answer generation
are untouched.

Existing LongMemEval traces can be analyzed without changing or rerunning the
benchmark core:

```bash
python PRAGMOS_analyze_trace_metrics.py \
  --trace path/to/the_longmemeval_trace.jsonl \
  --benchmark-family longmemeval \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --output path/to/trace_measurements.json
```

For legacy traces, the analyzer opens the GGUF in vocabulary-only mode solely
to count tokens. It performs no inference. New LoCoMo traces already contain
the exact counts, so they need only `--trace` and `--output`.

Context efficiency has two headline measurements:

1. Mean injected context tokens per question.
2. Full-source to injected-context token ratio. Higher means the source corpus
   was larger relative to the injected context. It must be interpreted jointly
   with retrieval coverage and answer quality; it does not alone prove that all
   source information was preserved.

Memory and graph quality has three headline measurements, all computed over
the evidence that could affect the answer:

1. Selected-memory provenance completeness.
2. Selected-graph provenance completeness.
3. Selected-graph temporal-field consistency.

Raw counts are retained only as audit denominators. Gold answers and evidence
are not used by these measurements. These three quality rates measure structural
trace integrity, not the semantic truth of extracted graph relationships;
retrieval and answer metrics cover downstream utility.

## Local Execution Claim

`--local-only` removes common cloud API credentials from the benchmark child
environment and sets Hugging Face and Transformers offline flags. The manifest
also records local model/data paths and `--n-gpu-layers`.

This supports the claim that inference used no hosted model, GPU/TPU, or hosted
storage. A run with `--n-gpu-layers 40` does use the Mac's local integrated
Metal GPU, so it must not be described as a CPU-only run.
