# PRAGMOS Synthetic LoCoMo Regression Suite 600 V1

This directory contains a deterministic 600-question subset of the independent
PRAGMOS Synthetic LoCoMo Development Set V1. Every complete source conversation
is retained; only the QA workload is reduced.

The suite is coverage-balanced for development and regression testing. It is
not an official LoCoMo split and its unweighted aggregate score is not an
estimate of official LoCoMo prevalence.

| Category | Task | Questions |
| --- | --- | ---: |
| 1 | Multi-hop | 80 |
| 2 | Temporal | 100 |
| 3 | Open-domain | 80 |
| 4 | Single-hop | 220 |
| 5 | Adversarial | 120 |
| **Total** |  | **600** |

All 37 synthetic families are represented. Selection first preserves evidence
complexity extrema, then spreads each family across all available fictional
speaker pairs, and finally fills the family quota using a stable hash rank.
Category 3 is deliberately oversampled and includes all seven available
multi-premise behavioral inference questions.

The source questions, answers, evidence IDs, and conversations are unchanged.
The manifest records source and output hashes, exact family targets, selection
policy, and validation results.

Regenerate and validate with:

```bash
python build_PRAGMOS_synthetic_LoCoMo_regression.py
python -m unittest test_build_PRAGMOS_synthetic_LoCoMo_regression.py
```

Report category and family metrics separately. Only untouched official LoCoMo
data may be used for official publication results.
