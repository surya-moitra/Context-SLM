# PRAGMOS Synthetic LongMemEval Development Set V1

This directory contains a deterministic, LongMemEval-shaped **development and
regression dataset**. It is not an official LongMemEval split and results on it
must not be reported as official LongMemEval scores.

## Construction policy

- The official 500-record file supplies only the ordered `question_type` profile
  and whether each record is an abstention example.
- Official questions, answers, entities, dates, and conversations are not used
  to generate synthetic content.
- Every synthetic record has fresh conversations, 38-62 sessions, answer-session
  provenance, hard near-entity distractors, and deterministic reference answers.
- The generator uses seed `20261001` and records file hashes in the manifest.
- The generated profile matches the official type counts and 30-case abstention
  distribution exactly.

The dataset contains 50 scenario families covering direct user facts,
assistant-generated lists and facts, preference constraints, two successive
knowledge updates, temporal calculations, counts, distinct counts, sums,
unit-bearing arithmetic, averages, differences, ratios, argmax/argmin questions,
and several kinds of incomplete-evidence abstention.

## Regenerate

```bash
.venv/bin/python generate_PRAGMOS_synthetic_LongMemEval.py
```

## Validate

```bash
.venv/bin/python -m unittest test_generate_PRAGMOS_synthetic_LongMemEval.py
```

Treat this set as frozen after beginning PRAGMOS improvements. New scenario
families or regenerated seeds must receive a new version and must not replace
V1 silently.

## Contiguous multi-session view

`pragmos_synthetic_longmemeval_500_v1_multisession_block.json` is an
order-only derivative of the frozen V1 dataset. All 500 records, IDs, and field
values are unchanged. The 133 `multi-session` records form one contiguous block
at zero-based indices `70-202` (positions `71-203` when counting from one).

Regenerate the order-only view and its provenance manifest with:

```bash
.venv/bin/python reorder_PRAGMOS_synthetic_LongMemEval.py
```

Validate both generation and reordering invariants with:

```bash
.venv/bin/python -m unittest test_generate_PRAGMOS_synthetic_LongMemEval.py test_reorder_PRAGMOS_synthetic_LongMemEval.py
```
