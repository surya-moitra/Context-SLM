# PRAGMOS Synthetic LoCoMo Development Set V1

This directory contains a deterministic, LoCoMo-shaped development and
regression dataset. It is not an official LoCoMo split, and results from it
must never be reported as official LoCoMo results.

## Frozen profile

The public LoCoMo repository at commit
`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` was inspected only to establish
the JSON schema and aggregate workload profile.

| Category | LoCoMo task | Questions |
| --- | --- | ---: |
| 1 | Multi-hop | 282 |
| 2 | Temporal | 321 |
| 3 | Open-domain | 96 |
| 4 | Single-hop | 841 |
| 5 | Adversarial | 446 |
| **Total** |  | **1,986** |

V1 contains ten independent fictional conversations, 19-32 sessions per
conversation, and exactly the same per-conversation category and turn-count
profile as the pinned release. The generated conversations contain 369-689
turns each.

Official conversations, turns, questions, answers, evidence text, speakers,
dates, observations, summaries, and events are not used as generation input.
An optional validation pass compares question strings with the official file;
V1 has zero exact question reuse.

## Stress coverage

- All 282 multi-hop questions require evidence from multiple sessions.
- Multi-hop families cover list union, place and collection aggregation,
  distinct counting, shared-speaker intersection, goals, recommendations, and
  8-19-item long-history unions.
- Temporal families cover yesterday, last week, next month, N days ago,
  explicit duration, previous weekday, year, exact date, and cross-session
  relative-date joins spanning up to four evidence turns.
- Open-domain families cover geographic, preference, career, gift, genre, tool,
  and behavioral inference, including synthesis across up to 17 premises.
- Adversarial families contain 301 wrong-speaker and 145 wrong-attribute
  near matches.
- 961 questions reference evidence turns carrying synthetic image captions;
  109 single-hop answers are available only from the caption.

## Files

- `pragmos_synthetic_locomo_1986_v1.json`: frozen synthetic dataset.
- `pragmos_synthetic_locomo_1986_v1_manifest.json`: hashes, provenance policy,
  profile, and validation results.
- `../../../generate_PRAGMOS_synthetic_LoCoMo.py`: deterministic generator.
- `../../../test_generate_PRAGMOS_synthetic_LoCoMo.py`: schema and provenance
  tests.

## Regenerate and validate

```bash
.venv/bin/python generate_PRAGMOS_synthetic_LoCoMo.py
.venv/bin/python -m unittest test_generate_PRAGMOS_synthetic_LoCoMo.py
```

To perform the optional no-reuse check without copying official data into this
repository:

```bash
.venv/bin/python generate_PRAGMOS_synthetic_LoCoMo.py \
  --official-data /path/to/LoCoMo/data/locomo10.json
```

## Freeze rule

V1 becomes immutable once the first PRAGMOS run begins. A genuinely necessary
correction must produce V2 with a new seed or generator version while retaining
V1 and its results. Synthetic performance is a development signal, not a paper
headline. Only untouched official LoCoMo data may produce the reported LoCoMo
score.
