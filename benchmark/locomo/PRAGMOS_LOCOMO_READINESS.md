# PRAGMOS LoCoMo Readiness Assessment

## Official protocol facts

The assessment uses the public LoCoMo release at commit
`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.

- The release contains ten conversations, 19-32 sessions per conversation,
  369-689 turns per conversation, and 1,986 QA items.
- Category 1 is multi-hop. The official scorer splits comma-separated answers
  and computes partial token F1. In the pinned data, 269 of 282 questions have
  evidence in multiple sessions; evidence count reaches 19.
- Category 2 is temporal. The official inference prompt appends an instruction
  to use conversation dates and return an approximate date.
- Category 3 is open-domain. It requires commonsense or external world
  knowledge in addition to retrieved conversation evidence.
- Category 4 is single-hop and uses token F1.
- Category 5 is adversarial. The official scorer awards credit only when output
  contains `no information available` or `not mentioned`.
- The official full-context renderer appends `blip_caption` when a turn contains
  one. Across the pinned QA set, 857 questions have at least one caption-bearing
  evidence turn.

## Frozen synthetic V1 coverage

The development set under `synthetic_v1` has 1,986 questions and exactly
matches the pinned release's per-conversation session counts, turn counts, and
category counts. It uses independent fictional content and has zero exact
question-string reuse against the pinned official file.

- All 282 synthetic multi-hop questions cross sessions; evidence depth reaches
  19 turns.
- Temporal questions include relative dates and genuine cross-session chains
  spanning up to four evidence turns.
- Open-domain questions reach 17 evidence turns and a mean evidence depth of
  2.14, close to the pinned release's 2.08.
- 961 questions cite caption-bearing evidence, including 109 caption-only
  single-hop answers.
- All 446 adversarial answers are null, with 301 wrong-speaker and 145
  wrong-attribute near matches.
- The manifest records the generator, dataset, and official validation-file
  SHA-256 hashes. Synthetic V1 becomes immutable when its first PRAGMOS run
  starts.

## Pre-freeze blockers

These are benchmark-adapter requirements, not evidence that the context graph
itself should be redesigned.

The existing LongMemEval runner is not a LoCoMo adapter and must not be pointed
at the LoCoMo-shaped JSON directly.

1. **Peer-speaker ingestion**: both speakers are factual subjects. Do not map
   one participant to the assistant-memory role. Preserve the named `speaker`
   while treating both as admissible factual evidence.
2. **Speaker-aware evidence text**: hard query anchors must see the source
   speaker even when the turn uses `I`. Render or score evidence as
   `<speaker>: <text>` while retaining the raw quote separately.
3. **Caption ingestion**: append the released `blip_caption` as explicitly typed
   visual evidence with the same dialog ID. The images themselves are absent.
4. **Category-specific prompting and scoring**: reproduce the pinned evaluator,
   especially the category-2 date instruction, category-1 comma-aware F1, and
   category-5 refusal rule.
5. **Canonical adversarial refusal**: use `No information available` for a
   confirmed LoCoMo category-5 abstention. `I do not know` is semantically sound
   but receives zero in the official implementation.
6. **Conversation isolation and reuse**: ingest each conversation once, answer
   all of its questions from the same immutable index, and clear state before
   the next conversation. Cache construction must not use QA evidence labels.

## Expected PRAGMOS failure areas

### High risk: complete multi-hop answers

The current retrieval defaults return too few items for questions whose gold
answer combines up to 19 evidence dialogs. Existing arithmetic covers counts,
sums, averages, and temporal joins, but LoCoMo also requires set union,
intersection, per-speaker grouping, and complete list recall. A correct partial
list still loses category-1 F1.

### High risk: adversarial near matches

Wrong-speaker and wrong-attribute questions deliberately retrieve highly
similar evidence. PRAGMOS has strong anchor and attribute safeguards, but they
must operate on named peer speakers. Without the adapter requirements above,
the model will either abstain on valid named-speaker questions or answer invalid
ones from the other speaker's turn.

### Medium-high risk: relative temporal arithmetic

PRAGMOS handles several temporal joins and date-indexed retrieval patterns, but
LoCoMo heavily uses `last week`, `last Friday`, `yesterday`, `N days ago`,
durations, and approximate dates relative to the session timestamp. Retrieval
can succeed while answer computation still fails.

### Medium risk: image-caption evidence

Ignoring captions makes some questions impossible. Treating captions as normal
untyped text can also contaminate speaker attribution. Caption evidence needs a
clear marker and the parent turn's provenance.

### Structurally model-limited: open-domain inference

Category 3 asks for likely careers, geographic mappings, holidays, suitable
gifts, preferences, and counterfactual judgments. PRAGMOS can retrieve the
premises but cannot guarantee that Phi-3 knows or reasons to the expected
answer. Report this category separately; do not disguise base-model knowledge
as a memory-layer failure or add benchmark-specific answer tables.

### Lower risk: direct single-hop recall

After speaker-aware ingestion, direct category-4 facts should be PRAGMOS's
strongest LoCoMo category. Remaining risks are long causal answers, exact list
phrasing, and caption-only details.

## Publication-safe sequence

1. Implement and test a LoCoMo adapter and official-compatible scorer without
   changing PRAGMOS retrieval behavior.
2. Commit and tag the synthetic dataset, generator, adapter, scorer, model hash,
   prompt, seed, and configuration.
3. Run small fixed slices from every synthetic family, then the frozen full
   synthetic V1 exactly once for final development validation.
4. Make only correctness fixes justified by family-level failures. Preserve all
   prior synthetic results and increment the dataset or code version.
5. Freeze code and configuration. Record the Git commit before opening official
   answers or running official evaluation.
6. Run raw Phi-3 and PRAGMOS on all ten official conversations with identical
   model and generation settings. Report overall and per-category F1, retrieval
   recall using evidence IDs, latency, context tokens, and ablations.
