# Opt-In Guarded Advice Integration

## Scope

Implemented in `/Users/moitra/.codex/worktrees/lme-evidence-compiler/context`.
Existing commands retain `--pragmos-assistance-mode off`. Enabling `guarded`
uses the normal benchmark entry point and writes actual predictions, not shadow
answers. There are no benchmark-label switches, question-ID rules, answer tables,
new weights or hosted calls. This is an experimental candidate, not a demonstrated
20/30 Preferences repair or an official benchmark result.

`PRAGMOS_context_layer_org.py` is unchanged, SHA-256:
`d993c6985c235fd9c6e038aee1034629f78e65b380e66c4d548cf65f7fd43491`.
The original observation-only boundary is also unchanged. The benchmark adapters
and execution argument module ARE changed to provide the opt-in integration.

## Behavior

1. One question-only, state-isolated local SLM call interprets the requested
   action. Only a complete, literal-quote-backed `create_advice` result enables
   circumstances collection. Recall, computation, ambiguity and invalid outputs
   retain the legacy answer path. This is model interpretation, not a semantic
   proof; unseen factual requests can still be misclassified.
2. Raw user-role sources are indexed, or existing prepared peer sources reused.
   Private retrieval views reuse the current dense/BM25/reranker stack. Whole
   source containers are restored with speaker, timestamp and literal offsets.
   Related qualifications/updates are selected and packed atomically.
3. A bounded source-mode interpretation distinguishes reported circumstances
   from imagined, quoted, task-only and uncertain sources. Acceptance requires a
   literal nonempty quote. The generated interpretation is diagnostic only:
   original paragraphs, not generated fact summaries, reach the answer model.
   All source interpretations together, including a retry, are capped at eight.
4. If the first attempt finds no usable personal context, one bounded local
   search reformulation may retry retrieval. Its two search phrases never become
   facts or answer context. Identical source interpretations are reused only
   within this one question; the total interpretation cap does not reset.
5. Whole-source packing enforces the actual 2,048-token model window, 64 output
   tokens, safety allowance and configured evidence cap. One final Phi-3 call
   writes the real advice. Missing context returns explicit uncertainty. A
   technical helper failure falls back to legacy; failed model-state restoration
   is fatal. No advice or interpreted fact is written into graph/vector memory.

The new path currently requires greedy decoding (`temperature=0`, `top_p=1`,
`repeat_penalty=1.1`). Unsupported ablations and simultaneous use of the frozen
observer are rejected explicitly. Other chat templates/models are not validated.

## Traces And Accounting

Both adapters hash the active helper, grammar and interpretation source files
and protocols in their manifests. Changing code/configuration requires a new run
name, not bypassing resume checks.

`execution_diagnostics.assistance_route` contains the request interpretation,
whether legacy was used, source interpretations, any search retry, literal
sources, packing decisions and the actual answer prompt. The source report's
`semantic_certification` and `semantic_answer_verified` remain false. The nested
component protocol describes defaults; the explicit `source_policy` and
`interpretation_policy` describe the executed policy. Guarded reports identify
their invocation as `opt_in_guarded_advice_route`.

End-to-end `elapsed_seconds` includes routing, indexing, scorer loading, source
interpretation, retry and generation. `llm_call_counts` includes actual request,
source, search and generation calls, including failed attempts. Component timings
are diagnostic and do not replace end-to-end latency. Gold labels are consulted
only after the answer is final to calculate retrieval metrics.

## Four Earlier Failures Retested

Native normal-runner results are preserved under:

```text
/Users/moitra/surya/dev_project/context/benchmark/longMemEval/synthetic_v2/semantic_revision_2/preferences_30_v1/guarded_validation_v1/smoke4_final
```

Phi-3 Mini Q4, seed 42, 2K context, 64 output tokens, greedy decoding, local CPU
retrieval. Same frozen R2 records and references as the earlier Preferences run.
Source IDs, categories and reference answers are not given to the advice model.

| Case | Actual answer behavior | End-to-end seconds | Assessment |
| --- | --- | ---: | --- |
| 0133, camera | Lightweight Tern-compatible tripod and hiking bag. | 42.39 | Better use of mount/weight, but low-light need not explicit; partial. |
| 0135, hotel | Explicit insufficient personal context. | 50.61 | Still fails: relevant room/view source missed, even after bounded reformulation. |
| 0144, party drinks | Non-alcoholic berry cordial with lemons and soda. | 38.03 | Manually consistent with the reference constraints. |
| 0160, commute | Audio stories, not reading while driving. | 104.28 | Manually consistent with the reference constraints. |

These are two clear manual improvements, one partial answer and one remaining
failure, NOT four semantic passes. Mean end-to-end time is 58.83 seconds for this
small selected sample, not a full-suite latency estimate. No GPT judge was run.
All final answer prompts plus output/safety reserves fit within 2K. The final
code adds only trace-label corrections after this native run; it does not change
the tested prompts, search, source decisions, packing or generation.

The hotel retry produced neutral searches for accommodation preferences and
lodging experiences, but source ranking still selected an unrelated tour request.
The commute trace also reveals source-classifier false positives on unrelated
recipe/instruction text; the answer model ignored those in this run. A literal
quote does NOT prove biography, relevance or completeness. Passing the controls
below does not remove these retrieval/interpretation limitations.

## Regression Checks

- Focused tests: 136 methods, 132 passed and four opt-in native checks skipped.
  Exact pre-integration adapter parity covers all six legacy families with the
  mode off. Guarded non-advice paths preserve answers, prompts, legacy calls,
  model state and memory; only routing diagnostics/call/latency accounting differ.
- Native controls: all 11 question-purpose controls routed as expected, including
  the previous observer false positive. Quoted-only and hypothetical-only sources
  produced no personalized answer. The resource-update control retained both
  statements and excluded the unavailable ingredient. Warmed token prefix, last
  logits, seed, independent factual completion and memory checks passed.
- Native control artifacts: `guarded_validation_v1/guard_controls.json` beside
  the four-question directory above. These provided-source fixtures are NOT
  extra benchmark accuracy passes.
- Full discovery: 1,309 tests, the SAME seven failure reports and one error as
  before integration, ten skips. The prior 1,287-test log has the same failing
  test/subtest identities: six older state-pipeline snapshot comparisons, the
  paused semantic-checker expectation, and frozen-publication-reference lookup.
  No old expectation, frozen hash or existing regression test was weakened.

These checks find no new tested legacy regression; they cannot guarantee zero
regression on all unseen requests. Keep this mode opt-in and review real answers.

## Run Preferences30

Preflight without inference:

```zsh
/bin/zsh /Users/moitra/.codex/worktrees/lme-evidence-compiler/context/run_PRAGMOS_preferences_guarded_30.zsh --check-only
```

Full run, with the absolute project interpreter and frozen dataset:

```zsh
caffeinate -i /bin/zsh \
  /Users/moitra/.codex/worktrees/lme-evidence-compiler/context/run_PRAGMOS_preferences_guarded_30.zsh
```

The runner enables guarded mode explicitly, keeps temporal/NLI/state experiments
off, validates the frozen R2 hash, uses a fresh `guarded_advice_v1` output directory,
resumes only compatible runs and preserves the console log plus standard files.
It does not run the GPT judge or alter any previous baseline.
