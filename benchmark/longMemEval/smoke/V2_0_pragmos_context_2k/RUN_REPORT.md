# PRAGMOS LongMemEval Smoke Benchmark V2.0

Date: 2026-09-19

## Scope

V2.0 is the first end-to-end LongMemEval smoke run using Phi-3 with the
PRAGMOS context layer. It uses the same 2,048-token model context window and
the same 512-token answer reserve as the V1.0 raw Phi-3 haystack baseline.

This is a five-example development smoke test, not a publishable LongMemEval
score. The first five dataset records were used during development, so future
claims must be evaluated on a larger untouched set with the official evaluator.

## Result

| Metric | V1.0 raw haystack | V2.0 PRAGMOS |
|---|---:|---:|
| Examples | 5 | 5 |
| Exact match | 0.000 | 0.000 |
| Contains reference | 0.000 | 0.600 |
| Average token F1 | 0.000 | 0.2927 |
| Answer-session retrieval recall | Not applicable | 1.000 |
| Average total time/example | 15.39s | 65.72s |

PRAGMOS beats the V1.0 score of zero under the same 2K context-window limit.
Three of five generated answers contain the complete reference answer. All
five answer-bearing sessions were retrieved, which separates the two remaining
failures from basic retrieval misses.

## Per-Example Results

| Question | Reference | Prediction | Contains reference |
|---|---|---|---:|
| What degree did I graduate with? | Business Administration | You graduated with a degree in Business Administration. | 1 |
| How long is my daily commute to work? | 45 minutes each way | Your daily commute to work takes 1 hour each way. | 0 |
| Where did I redeem a $5 coupon on coffee creamer? | Target | You redeemed a $5 coupon on coffee creamer last Sunday. | 0 |
| What play did I attend at the local community theater? | The Glass Menagerie | The play you attended at the local community theater was "The Glass Menagerie." | 1 |
| What is the name of the playlist I created on Spotify? | Summer Vibes | You mentioned the playlist called Summer Vibes on Spotify. | 1 |

Exact match remains zero because Phi-3 returns short sentences rather than
only the reference span. For this smoke test, `contains_reference` is easier to
interpret than exact match. Token F1 also gives partial credit to the incorrect
commute answer because it shares words with the reference, so it must not be
treated as semantic correctness.

## Method

For each LongMemEval record, the benchmark:

1. Resets all PRAGMOS graph, vector, lexical, entity, and conversation state.
2. Converts every haystack message into a structured turn with its original
   session ID, role, speaker, session date, text, and a unique turn ID.
3. Indexes all raw turns in overlapping 160-word chunks with 32-word overlap.
   Long turns are chunked before embedding so MiniLM does not silently discard
   evidence beyond its input limit.
4. Retrieves candidates with normalized dense embeddings and BM25.
5. Reranks locally with `cross-encoder/ms-marco-MiniLM-L6-v2`.
6. Materializes graph relations from the four strongest raw-turn candidates,
   traverses the graph in both directions to depth two, and reruns hybrid
   retrieval with graph expansion.
7. Adds a bounded two-turn same-session neighborhood for cross-turn evidence.
8. Builds an evidence-labeled prompt under a dynamically measured hard token
   budget, then asks the same local Phi-3 GGUF to answer.

The benchmark never uses message-level `has_answer` annotations for indexing,
retrieval, graph construction, reranking, prompt creation, or generation. The
dataset's `answer_session_ids` are used only after generation to calculate
retrieval diagnostics.

## Configuration

- Hardware: Apple A18 Pro MacBook Neo, 8 GB unified memory
- Python: 3.11.16
- Generator and graph extractor:
  `Phi-3-mini-4k-instruct-q4.gguf`
- Context window: 2,048 tokens
- Maximum completion: 512 tokens
- Temperature: 0.0
- GPU layers: 40
- Dense embedder: `all-MiniLM-L6-v2`
- Local reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Retrieval top-k: 6
- Retrieval minimum score: 0.20
- Graph candidates: 4
- Graph traversal depth: 2
- Graph evidence cap: 4
- Session-neighbor radius: 2 turns
- Session-neighbor cap: 4

Prompt estimates ranged from 1,321 to 1,472 tokens, leaving the configured
512-token completion reserve and wrapper safety space inside `n_ctx=2048`.

## Runtime

- Total: 328.61 seconds
- Average ingestion: 6.83 seconds/example
- Average retrieval, reranking, and graph work: 45.44 seconds/example
- Average answer generation: 13.40 seconds/example
- Average end-to-end: 65.72 seconds/example

Each example ingested 485-616 turns and created 869-960 memory records. This
is materially slower than raw haystack stuffing, as expected: PRAGMOS performs
full-haystack indexing, local reranking, and selective graph extraction.

## Failure Analysis

### Commute Duration

Retrieval found the correct user statement: "my daily commute ... takes 45
minutes each way." The same session also contains an assistant response that
incorrectly paraphrases this as "an hour each way." Phi-3 selected the
assistant-generated contamination despite role labels and the instruction to
prefer explicit user statements.

This is a memory trust and contradiction-resolution problem. The next design
iteration should assign source authority by role and detect unsupported
assistant restatements rather than storing all turns with equal evidentiary
weight.

### Coupon Location

Retrieval found the answer-bearing session and neighboring turns that mention
Target. The coupon statement itself specifies the item and date but omits the
store; Target must be inferred from nearby discussion of the Target Cartwheel
app. Phi-3 answered the explicit date instead of resolving the implicit place.

This is a cross-turn evidence synthesis failure. Adding answer-specific rules
would overfit this five-example development slice, so V2.0 records the miss.

## Interpretation

The first result supports the basic PRAGMOS hypothesis: full-haystack memory
retrieval can recover useful evidence that raw recent-session stuffing cannot
fit into a 2K prompt. It does not yet establish benchmark superiority at
publication quality.

The strongest diagnostic is the gap between answer-session retrieval recall
(`1.00`) and answer containment (`0.60`). Retrieval is working on this sample;
source trust, contradiction handling, and small-model evidence synthesis are
now the dominant errors.

## Command

```bash
python PRAGMOS_benchmark_LongMemEval.py \
  --data-file benchmark/longMemEval/longmemeval_s_cleaned.json \
  --mode pragmos_context \
  --limit 5 \
  --output-dir benchmark/longMemEval/V2_0_pragmos_context_2k \
  --run-name pragmos_context_v2_0_smoke \
  --model-path ../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --n-ctx 2048
```

## Files

- `pragmos_context_v2_0_smoke_predictions.jsonl`: evaluator-style hypotheses.
- `pragmos_context_v2_0_smoke_trace.jsonl`: full per-record retrieval,
  provenance, graph, timing, prompt-token, and diagnostic trace.
- `pragmos_context_v2_0_smoke_summary.json`: aggregate settings and metrics.
- `diagnostic_pre_neighbor/`: an earlier development run retained for audit;
  it is not the canonical V2.0 result.

