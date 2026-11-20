# Isolated Circumstance Pipeline

This document records the earlier literal-source component experiment. Its
standalone defaults remain available. The later **opt-in guarded integration**,
new source-interpretation policy, current checks and known failures are described
in `PRAGMOS_ASSISTANCE_INTEGRATION.md`. Historical statements below about no
adapter integration describe that earlier stage, not the current guarded mode.

## Status

Implemented as a standalone component experiment, not a production repair or an
automatic route. **Do not enable it as the Preferences answerer yet.** Native
inspection of three improved haystack cases does not establish the general
applicability or automatic-routing acceptance gate.

No existing inference module was edited during this implementation. The core,
both benchmark adapters, existing execution controls and observation boundary
retain their previous hashes. Existing commands do not invoke the new helper.
No recommendation or interpreted personal fact is written into memory.

## Implementation

`PRAGMOS_assistance_context.py` and `PRAGMOS_assistance_selection.py` provide five operations:

1. `retrieve_assistance_evidence`: creates a private view of raw memories from
   explicitly identified source speakers. It filters before top-k, reconstructs
   existing normalized FAISS vectors without re-embedding the haystack, rebuilds
   lexical statistics for that view and reuses local dense/BM25/reranking models.
   Graph expansion is disabled. Two bounded, generic queries are used; results
   are thresholded and deduplicated by provenance. Shared caches are not mutated.
2. `build_circumstance_bundles`: validates raw-source identity and quote offsets,
   restores whole source containers and retains qualifications together. Duplicate
   chunks merge, but different speakers, sessions, turns or timestamps do not.
   Only an explicit, unambiguous same-speaker/session parent may add linked
   context. That context is not converted into a new preference.
3. `rank_circumstance_bundles`: rescores the restored sources, not their original
   retrieval chunks, with the existing local cross-encoder. Long sources are
   covered by overlapping, token-checked windows, with at most 128 scoring pairs.
   The best window score ranks each complete source. A minimal score-ranked
   prefix retains 90% of the softmax relative score weight by default. This is a
   utility budget, NOT a calibrated probability or an assertion/ownership check.
   `--relevance-mass 1` retains all completely scored sources for an ablation.
   A second, bounded source-to-source pass retains same-speaker related sources
   as potential qualifications/updates. It uses the local scorer's raw score
   above zero, not a claim that the texts express the same fact. There are at most
   128 such pairs, with one hop only. No fact is overwritten or interpreted by
   this step. Ranking caches/counters remain private; no extra Phi-3 call occurs.
   Missing scorers, incomplete work, omitted sources and scores remain explicit
   in the trace. An unavailable scorer does not silently fall back to chunk scores.
4. `pack_assistance_context`: packs complete bundles within the actual tokenizer's
   2,048-token context, output reserve and safety allowance. Oversized bundles are
   skipped whole and recorded; no source or question is prefix-truncated. Original
   IDs and offsets remain in the trace. The answering-SLM prompt uses opaque
   session/turn aliases so synthetic authoring labels are not exposed as hints.
   Sources linked as potential qualifications are packed together or skipped
   together, never as a partial state. This does not prove that every relevant
   qualification was retrieved or linked.
5. `run_isolated_assistance`: requires an explicit `AssistanceRequest`, not an
   observed ASSIST label. The same local SLM receives the raw circumstances and
   formulates advice in one generation call. It restores model state, including
   on exceptions/interruption, and refuses external model caches. Failed state
   restoration is fatal. Token-limited free text is retained with a warning, not
   certified complete or silently discarded.

Raw user-authored dialogue is not automatically a personal fact. The prompt
distinguishes quotations, fiction, requests, suggestions, resource availability,
preferences, owners and dated changes. These are instructions to the model,
**not deterministic semantic guarantees**. No new question regex, benchmark
category switch, answer lookup table, NLI checker, hosted service or model weights
were added. This version has been tested with Phi-3; other chat templates have
not been validated.

## Source-Selection Verification (v5)

The implemented source selector does NOT use the attempted generative source
classifier. Native preflights showed both false positives (unrelated documents
and hypothetical text) and false negatives (explicit personal circumstances).
Literal-quote checking verifies a substring, not its semantic applicability.
Those classifier prototypes were removed, not used as a new production gate.

The source-only ranking trial improved the camera and drink full-haystack
answers. It also exposed two limits: the commute response still adopted a
transport-presentation suggestion, and an independent update fixture lost a
later source under the query-score cut. The related-source/atomic-packing stage
addresses the second structural omission. It does not solve the first semantic
assertion/ownership problem. Fiction/hypothetical controls remain acceptance
tests to review, not assumed passes from syntactically valid outputs.

Final probe artifacts, including code/model/dataset hashes:

```text
/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2/preferences_30_v1/isolated_circumstance_source_selection_v2
```

Manual review of the final three actual full-haystack outputs:

| Case | Actual behavior | Prompt tokens, v4 -> v5 |
| --- | --- | --- |
| Camera | Lightweight Tern-compatible support and a hiking bag; no unrelated tutorial/article answer. | 1,965 -> 375 |
| Party drink | Non-alcoholic berry cordial with the supplied lemons and soda water. | 1,907 -> 400 |
| Commute | Audio stories/podcasts; no recommendation to change transportation. | 1,898 -> 1,214 |

These are three manually reasonable answers, not judge-scored benchmark passes.
All three generations ended normally within 64 tokens. The helper used one Phi-3
generation call per case, with component times of 12.44, 11.18 and 18.79 seconds.
Those times exclude indexing, model loading and harness controls. Source selection
itself took 0.93, 0.63 and 1.16 seconds respectively. Model state, memory/cache
fingerprints and the separate legacy factual answer were unchanged in all three.

The probe's `source_selection` records query-selected sources, potential
qualification links, window scores, work omissions and atomic packing groups.
Raw source text, full prompts and actual generated answers remain available.
No production router, core, adapter or existing execution mode imports this
selector. It remains an explicitly invoked component experiment.

Nine provided-source native scenarios were also inspected, with reports in
`native_controls/` under that probe directory. The three original clean
counterfactuals remain context-sensitive. Noisy weekend/dinner advice and the
unknown related-owner case were reasonable on manual inspection. The resource
update control now retains both statements atomically and excludes the unavailable
ingredient from the recommendation. **Quoted-only and hypothetical-only sources
still influence advice without establishing actual personal circumstances.**
Those are unresolved limitations, not semantic passes. Test-method success in
these controls certifies state/provenance mechanics only, not answer accuracy.

The tokenizer may warn about long sequences while constructing complete offset
maps. Those long sequences are not passed to the cross-encoder: each actual
scoring pair is explicitly checked against its token limit and traced.

## Earlier Native Baseline (v4)

The final real-haystack component probe is stored at:

```text
/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2/preferences_30_v1/isolated_circumstance_probe_final_v2
```

It records model/data/code hashes, raw generated text, the full answer prompts,
source spans, packing omissions, call counts and isolation checks. It uses the
actual R2 synthetic haystacks, without authoring audit addresses or gold answers
as inference inputs. The caller explicitly selects assistance test cases; this
does not test automatic purpose detection. References are recorded only after
inference. The audit was consulted separately AFTER inference to check source
retention.

Configuration: Phi-3 Mini Q4, n_ctx=2048, max_tokens=64, seed=42, temperature=0,
top_p=1, repeat_penalty=1.1, threads=8, GPU layers=40; local CPU retrieval.

| Full-haystack case | Manual assessment |
| --- | --- |
| 0133, camera | Fails complete personalization: generic accessories; missing the supplied mount, weight and low-light requirements. |
| 0144, party drink | Fails: mint/lime cocktail advice instead of the supported alcohol-free cordial/lemons/soda combination. |
| 0160, commute | Fails: recommends changing transport rather than an activity consistent with driving and audio preferences. |

All three intended supporting paragraphs were packed in full. Prompts used
1,965, 1,907 and 1,898 tokens respectively. All outputs reached the 64-token
allowance and are explicitly marked output-limited. This is not an excuse for
the semantic failures; the drink/commute outputs already take the wrong direction.
Memory, graph/vector/cache fingerprints, warmed model prefix/logits/seed and a
separate legacy factual completion were unchanged in all three probes.

Helper elapsed times were 25.34, 30.21 and 62.23 seconds, excluding indexing
(9.57, 12.75 and 94.18 seconds), model loading and harness controls. These are
component diagnostics, not an official end-to-end PRAGMOS latency claim. There
was one generation call per question and no graph-materialization call.

A separate native clean-source test used counterfactual fixtures, not gold
benchmark sources. It deliberately used min_score=0 to retain the sole provided
source and isolate generation from retrieval thresholding:

| Changed circumstance | Actual answer behavior |
| --- | --- |
| Moss mount, daylight hikes, no heavy equipment | Suggested lightweight Moss-compatible support. |
| Pear juice/sparkling water, no alcohol, dislike sweetness | Suggested a non-alcoholic, less-sweet pear/sparkling-water drink. |
| Train journey, enjoys reading, dislikes audio | Suggested reading, not the original driving/audio answer. |

All three clean outputs ended normally in 30-35 tokens, with 355-363-token
prompts; component times were 4.75-6.00 seconds. Their source/prompt/output traces
are in `clean_counterfactual_controls.json` beside the final probe artifacts.
These manually reasonable answers are **not three benchmark accuracy passes**.
No GPT judge was called. The counterfactuals demonstrate sensitivity to supplied
facts, not high-recall/high-precision retrieval on long conversations.

Earlier trial directories are retained. The first multi-block source-review
classifier rejected useful sources in all three cases; it was removed rather
than promoted as a trustworthy gate. Dense JSON framing also encouraged request
echoing. The final helper has no intermediate fact classifier and uses readable
source blocks with a final plain-text request.

## Checks And Limits

- 43 offline component tests cover provenance, owners, explicit parents, atomic
  packing, opaque IDs, token-checked whole-source windows, relative score allocation,
  bounded qualification links, incompatible metrics and isolation/error paths.
- Focused checks also preserve the existing six-family behavior and prior
  source-projection/proof-first guards: 111 passed, three native-only skips.
- Full discovery ran 1,287 tests, with the same seven prior failure reports and
  one error: six older snapshot comparisons, a paused semantic-checker case and
  the frozen-publication-reference mismatch. Nine were skipped. No unrelated
  expectation was weakened. The two native control methods passed their
  structural/isolation assertions; their semantic limitations are recorded above.
- No official dataset inference, full Preferences30 run, GPT judge, new model
  download, automatic routing or production promotion occurred.

The acceptance gate remains semantic applicability and reliable purpose routing,
not merely smaller prompts. Query-relative ranking can remove distractors but
cannot certify personal ownership, adoption, or fictional vs actual context.
Neither the earlier nor the new isolated results justify a 20/30 Preferences
claim, a general accuracy projection, or automatic promotion.

## Commands

Offline checks, from the worktree:

```zsh
cd /Users/moitra/.codex/worktrees/lme-evidence-compiler/context
PY=/Users/moitra/surya/dev_project/context/.venv/bin/python
MPLCONFIGDIR=/tmp/pragmos-matplotlib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -B -m unittest test_PRAGMOS_assistance_selection test_PRAGMOS_assistance_context \
  test_PRAGMOS_assistance_boundary test_PRAGMOS_source_projection \
  test_PRAGMOS_execution test_PRAGMOS_proof_execution
```

Explicit native counterfactual controls:

```zsh
PRAGMOS_ASSISTANCE_CONTEXT_NATIVE_MODEL=/Users/moitra/surya/dev_project/phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  "$PY" -B -m unittest -v test_PRAGMOS_assistance_context.NativeCircumstanceControls
```

`probe_PRAGMOS_assistance_context.py --help` describes the real-haystack component
probe. Its output directory must be new; it never overwrites an existing run.
Do not use this probe as the official benchmark runner.
