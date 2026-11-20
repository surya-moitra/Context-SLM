"""Query-relative ranking for the isolated assistance experiment, not a fact gate."""

from collections import OrderedDict
import copy
import math


DEFAULT_RELEVANCE_MASS = 0.90
MAX_RANK_WINDOWS = 128
MAX_PAIR_TOKENS = 512
MAX_COHESION_PAIRS = 128
COHESION_MIN_RAW_SCORE = 0.0


def selection_identity():
    return {"policy": "restored_source_rerank_relative_mass_prefix_v1",
            "default_relative_mass": DEFAULT_RELEVANCE_MASS, "max_rank_windows": MAX_RANK_WINDOWS,
            "max_pair_tokens": MAX_PAIR_TOKENS, "calibrated_probability": False,
            "related_source_min_raw_score": COHESION_MIN_RAW_SCORE,
            "max_cohesion_pairs": MAX_COHESION_PAIRS,
            "personal_applicability_certified": False, "generative_review_calls": 0}


def _source_text(bundle):
    parts = [p.quote for p in bundle.paragraphs]
    parts.extend(p.quote for p in bundle.linked_context)
    return "\n\n".join(parts)


def _windows(tokenizer, question, text, pair_limit):
    """Cover the entire restored text; the scorer must never silently truncate it."""
    capacity = pair_limit - len(tokenizer.encode(question, add_special_tokens=False))
    capacity -= tokenizer.num_special_tokens_to_add(pair=True) + 8
    if capacity < 8:
        raise ValueError("question_exceeds_reranker_window")
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        raise ValueError("source_has_no_reranker_tokens")
    start, windows = 0, []
    while start < len(offsets):
        end = min(len(offsets), start + capacity)
        while end > start:
            first, last = offsets[start][0], offsets[end - 1][1]
            fragment = text[first:last]
            pair_tokens = len(tokenizer(question, fragment, truncation=False)["input_ids"])
            if pair_tokens <= pair_limit:
                break
            end -= 1
        if end <= start:
            raise ValueError("source_token_exceeds_reranker_window")
        windows.append((first, last, fragment, pair_tokens))
        if len(windows) > MAX_RANK_WINDOWS:
            raise ValueError("whole_source_exceeds_ranking_work_budget")
        if end == len(offsets):
            break
        start = end - min(32, (end - start) // 4)
    return windows


def relevance_prefix(scores, mass=DEFAULT_RELEVANCE_MASS):
    """Relative utility allocation, NOT probability of relevance or factual truth."""
    if isinstance(mass, bool) or not math.isfinite(mass) or not 0 < mass <= 1:
        raise ValueError("relative_relevance_mass_must_be_in_zero_one")
    if not scores:
        return [], []
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("finite_source_relevance_scores_required")
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    selected, cumulative = [], 0.0
    for i in order:
        selected.append(i)
        cumulative += weights[i]
        if mass < 1 and cumulative >= mass:
            break
    return selected, weights


def _related_closure(view, scorer, bundles, seed_ids, audit, pair_limit):
    """Keep potential qualifications with their source, not just high-query-score text."""
    links = audit["related_sources"] = {"pairs": 0, "links": [], "unscored": [],
                                        "semantically_certified": False}
    by_id = {b.bundle_id: b for b in bundles}
    pairs, owners = [], []
    # Only one hop from query-selected seeds, with a separate hard work bound.
    for seed_id in seed_ids:
        source = by_id[seed_id]
        for other in bundles:
            if other.bundle_id in seed_ids or (source.speaker, source.role) != (other.speaker, other.role):
                continue
            try:
                left = _windows(scorer.tokenizer, "", _source_text(source), pair_limit // 2)
                right = _windows(scorer.tokenizer, "", _source_text(other), pair_limit // 2)
                if len(pairs) + len(left) * len(right) > MAX_COHESION_PAIRS:
                    raise ValueError("whole_relation_exceeds_cohesion_work_budget")
            except (ValueError, TypeError, KeyError) as exc:
                links["unscored"].append({"source": seed_id, "target": other.bundle_id, "reason": str(exc)})
                continue
            for _a, _b, query, _count in left:
                for _c, _d, text, _other_count in right:
                    if len(scorer.tokenizer(query, text, truncation=False)["input_ids"]) > pair_limit:
                        raise ValueError("cohesion_pair_exceeds_reranker_window")
                    pairs.append((query, text))
                    owners.append((seed_id, other.bundle_id))
    links["pairs"] = len(pairs)
    values = view.predict_reranker_pairs(pairs) if pairs else []
    if values is None or len(values) != len(pairs) or any(not math.isfinite(float(s)) for s in values):
        raise ValueError("source_cohesion_scoring_failed")
    scores = {}
    for pair, score in zip(owners, values):
        scores[pair] = max(scores.get(pair, -math.inf), float(score))
    groups = [{key} for key in seed_ids]
    for (source, target), score in scores.items():
        if score <= COHESION_MIN_RAW_SCORE:
            continue
        links["links"].append({"source": source, "target": target, "raw_relatedness_score": score})
        group = {source, target}
        touching = [g for g in groups if g.intersection(group)]
        for existing in touching:
            group.update(existing)
            groups.remove(existing)
        groups.append(group)
    original_order = {b.bundle_id: i for i, b in enumerate(bundles)}
    audit["packing_groups"] = [sorted(group, key=original_order.get) for group in groups if len(group) > 1]
    selected = list(seed_ids)
    selected.extend(b.bundle_id for b in bundles if any(b.bundle_id in group for group in groups)
                    and b.bundle_id not in seed_ids)
    return selected


def rank_circumstance_bundles(context, request, bundles, *, relevance_mass=DEFAULT_RELEVANCE_MASS):
    """Score literal source windows on a private cache view; keep whole bundles."""
    relevance_prefix([], relevance_mass)
    audit = {**selection_identity(), "relative_mass": relevance_mass, "sources": [],
             "selected_ids": [], "omitted_ids": [], "unscored_ids": [], "scoring_pairs": 0,
             "status": "not_run", "elapsed_seconds": 0.0}
    view = copy.copy(context)
    view.reranker_score_cache = OrderedDict(context.reranker_score_cache)
    view.cache_counters = dict(context.cache_counters)
    scorer = view.get_local_reranker()
    if scorer is None or not callable(getattr(scorer, "tokenizer", None)):
        audit["status"] = "local_source_reranker_unavailable"
        audit["unscored_ids"] = [b.bundle_id for b in bundles]
        return (), audit
    pair_limit = min(MAX_PAIR_TOKENS, int(scorer.max_length or MAX_PAIR_TOKENS))
    pairs, owners, candidates = [], [], []
    # Short complete sources first: a giant document cannot consume all scoring work.
    for bundle in sorted(bundles, key=lambda b: len(_source_text(b))):
        item = {"bundle_id": bundle.bundle_id, "windows": []}
        audit["sources"].append(item)
        try:
            queries = list(dict.fromkeys([request.question, *request.retrieval_queries]))
            windows = [(query, window) for query in queries
                       for window in _windows(scorer.tokenizer, query, _source_text(bundle), pair_limit)]
            if len(pairs) + len(windows) > MAX_RANK_WINDOWS:
                raise ValueError("whole_source_exceeds_ranking_work_budget")
        except (ValueError, TypeError, KeyError) as exc:
            item.update(status="unscored", reason=str(exc))
            audit["unscored_ids"].append(bundle.bundle_id)
            continue
        index = len(candidates)
        candidates.append((bundle, item))
        for query, (first, last, fragment, count) in windows:
            item["windows"].append({"restored_text_start": first, "restored_text_end": last,
                                    "pair_tokens": count, "query": query})
            pairs.append((query, fragment))
            owners.append(index)
    audit["scoring_pairs"] = len(pairs)
    if not pairs:
        audit["status"] = "no_completely_rankable_sources"
        return (), audit
    scores = view.predict_reranker_pairs(pairs)
    if scores is None or len(scores) != len(pairs) or any(not math.isfinite(float(s)) for s in scores):
        audit["status"] = "source_reranking_failed"
        audit["unscored_ids"] = [b.bundle_id for b in bundles]
        return (), audit
    per_source = [-math.inf] * len(candidates)
    positions = [0] * len(candidates)
    for owner, score in zip(owners, scores):
        score = float(score)
        per_source[owner] = max(per_source[owner], score)
        candidates[owner][1]["windows"][positions[owner]]["raw_relevance_score"] = score
        positions[owner] += 1
    original = {b.bundle_id: i for i, b in enumerate(bundles)}
    order = sorted(range(len(candidates)), key=lambda i: original[candidates[i][0].bundle_id])
    selected_order, ordered_weights = relevance_prefix([per_source[i] for i in order], relevance_mass)
    selected = [order[i] for i in selected_order]
    weights = dict(zip(order, ordered_weights))
    for i, (_bundle, item) in enumerate(candidates):
        item.update(status="selected" if i in selected else "omitted_by_relative_utility_budget",
                    raw_relevance_score=per_source[i], relative_relevance_weight=weights[i])
    seed_ids = [candidates[i][0].bundle_id for i in selected]
    audit["query_selected_ids"] = seed_ids
    selected_ids = _related_closure(view, scorer, bundles, seed_ids, audit, pair_limit)
    selected_bundles = tuple(next(b for b in bundles if b.bundle_id == key) for key in selected_ids)
    for _bundle, item in candidates:
        if item["bundle_id"] in selected_ids and item["bundle_id"] not in seed_ids:
            item["status"] = "retained_as_related_source_not_an_interpreted_update"
    audit["selected_ids"] = [b.bundle_id for b in selected_bundles]
    audit["omitted_ids"] = [b.bundle_id for b, _ in candidates if b.bundle_id not in audit["selected_ids"]]
    audit["status"] = "ranked_not_semantically_certified"
    return selected_bundles, audit
