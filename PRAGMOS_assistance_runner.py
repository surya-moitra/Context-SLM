"""Advice-result serialization shared by benchmark adapters; no dataset labels."""

import copy

from PRAGMOS_assistance_context import AssistanceRequest, MAX_SOURCE_INTERPRETATIONS, run_isolated_assistance, protocol_identity as context_identity
from PRAGMOS_assistance_contracts import expand_memory_search, protocol_identity as contract_identity


SOURCE_FILES = (
    "PRAGMOS_assistance_runner.py", "PRAGMOS_assistance_context.py", "PRAGMOS_assistance_contracts.py",
    "PRAGMOS_assistance_selection.py", "PRAGMOS_assistance_boundary.py", "PRAGMOS_composable_candidate.py",
    "PRAGMOS_composable_execution.py",
)
INSUFFICIENT_CONTEXT = "I don't have reliable information about your actual circumstances to personalize that advice."


def protocol_identity():
    return {"protocol": "guarded-advice-route-v1", "default_enabled": False,
            "category_labels_used": False, "interpretation": contract_identity(),
            "circumstances": context_identity(), "source_policy": "interpreted",
            "search_retries": 1, "max_total_source_interpretations": MAX_SOURCE_INTERPRETATIONS}


def source_rows(context, bundles):
    """Describe only literal source spans that the answer actually received."""
    turns = {str(t.turn_id): t for t in context.raw_source_turns.values()}
    rows, seen = [], set()
    for bundle in bundles:
        sources = [(bundle["turn_id"], p["start"], p["end"], p["quote"]) for p in bundle["paragraphs"]]
        sources.extend((p["turn_id"], 0, len(p["quote"]), p["quote"]) for p in bundle.get("linked_context", ()))
        for turn_id, start, end, quote in sources:
            turn = turns[str(turn_id)]
            if turn.text[start:end] != quote:
                raise ValueError("assistance_trace_quote_not_authoritative")
            key = (str(turn_id), start, end)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"label": "raw_turn_chunk", "text": quote, "source_quote": quote,
                         "source_turn_ids": [turn.turn_id], "source_session_id": turn.session_id,
                         "source_start": start, "source_end": end, "role": turn.role,
                         "speaker": turn.speaker, "timestamp": turn.timestamp,
                         "external_turn_id": getattr(turn, "external_turn_id", None),
                         "parent_turn_id": getattr(turn, "parent_turn_id", None),
                         "evidence_type": getattr(turn, "evidence_type", None),
                         "score": bundle["score"], "retrieval_kind": "assistance_context"})
    return rows


def run_advice(context, question, args, turns):
    speakers = tuple(dict.fromkeys(t.speaker for t in turns if t.role.strip().casefold() == "user"))
    if not 1 <= len(speakers) <= 4:
        return None, {"status": "unsupported_source_speakers", "model_calls": {"total": 0}}
    # Share loaded weights across private retrieval/ranking views, never memory writes.
    context = copy.copy(context)
    context.get_local_reranker()
    cache = {}
    def attempt(queries=(), remaining=MAX_SOURCE_INTERPRETATIONS):
        return run_isolated_assistance(
            context, AssistanceRequest(question, speakers, retrieval_queries=queries), n_ctx=args.n_ctx,
            max_tokens=args.max_tokens, seed=args.seed, top_k=12, min_score=args.pragmos_min_score,
            source_policy="interpreted", evidence_token_budget=args.pragmos_answer_context_tokens,
            interpretation_cache=cache, max_interpretation_calls=remaining)
    report = attempt()
    if report["status"] in {"no_source_context", "no_disclosed_personal_context"}:
        first = report
        expansion = expand_memory_search(context, question, n_ctx=args.n_ctx, seed=args.seed)
        queries = tuple(dict.fromkeys(q.strip() for q in (expansion.get("payload") or {}).get("queries", ())
                                     if q.strip() and q.strip() != question))
        remaining = MAX_SOURCE_INTERPRETATIONS - first["model_calls"].get("source_interpretation", 0)
        retried = expansion["status"] == "interpreted" and bool(queries) and remaining > 0
        report = attempt(queries, remaining) if retried else dict(first)
        report["search_retry"] = {"first_attempt": first, "query_expansion": expansion, "retried": retried,
                                  "queries_are_search_terms_not_evidence": True}
        if retried:
            report["model_calls"] = {key: first["model_calls"].get(key, 0) + report["model_calls"].get(key, 0)
                                     for key in set(first["model_calls"]) | set(report["model_calls"])}
            report["phase_seconds"] = {key: first.get("phase_seconds", {}).get(key, 0) + report.get("phase_seconds", {}).get(key, 0)
                                       for key in set(first.get("phase_seconds", {})) | set(report.get("phase_seconds", {}))}
        else:
            report["model_calls"] = dict(report["model_calls"])
        report["model_calls"]["search_reformulation"] = expansion["model_calls"]
        report["model_calls"]["total"] = sum(v for k, v in report["model_calls"].items() if k != "total")
    report.update(invocation="opt_in_guarded_advice_route", automatic_routing_enabled=True)
    if report["status"] not in {"generated_candidate", "generated_candidate_output_limited",
                                 "no_source_context", "no_disclosed_personal_context"}:
        return None, report
    generated = report["model_calls"]["generation"] == 1 and bool(report.get("answer"))
    answer = report["answer"] if generated else INSUFFICIENT_CONTEXT
    final = source_rows(context, report.get("selected_bundles", ())) if generated else []
    candidates = source_rows(context, report.get("retrieved_bundles", ()))
    phases = report.get("phase_seconds", {})
    skipped = {"status": "not_run", "reason": "new_advice_route"}
    operation = {"operation": "none", "requires_session_diversity": False, "minimum_sessions": 1}
    packing = report.get("answer_packing", {})
    output = report.get("model_outputs", [])
    calls = {"semantic_query_planner": 0, "graph_materialization": 0, "candidate_extraction": 0,
             "final_generation": int(generated), "assistance_source_interpretation":
                 report["model_calls"].get("source_interpretation", 0),
             "assistance_search_reformulation": report["model_calls"].get("search_reformulation", 0)}
    calls["total"] = sum(calls.values())
    result = {"text": answer, "answer_decision": "contextual_assistance" if generated else "assistance_insufficient_context",
        "safe_abstention": not generated, "abstention_reason": None if generated else report["status"],
        "baseline_prediction": None, "baseline_answer_decision": "not_run_new_advice_route",
        "raw_generation_prediction": report.get("answer"),
        "prompt_tokens_estimate": packing.get("prompt_tokens", 0) if generated else 0,
        "completion_tokens_estimate": context.count_tokens(answer),
        "context_tokens_estimate": packing.get("source_context_tokens", 0) if generated else 0,
        "context_token_budget": min(args.n_ctx, args.pragmos_answer_context_tokens, 2048),
        "context_packing_diagnostics": {**packing, "final_answer_prompt_used": generated,
            "included_units": [{**row, "source_integrity": "authoritative_turn"} for row in final]},
        "generation_diagnostics": {"model_generation_used": generated,
            "finish_reason": output[-1].get("finish_reason") if output else None,
            "output_limit_reached": report.get("output_limit_reached", False)},
        "retrieval_seconds": phases.get("retrieval", 0) + phases.get("source_ranking", 0),
        "initial_retrieval_seconds": phases.get("retrieval", 0),
        "preference_synthesis_seconds": phases.get("source_interpretation", 0),
        "context_build_seconds": phases.get("source_restoration", 0) + phases.get("packing", 0),
        "generation_seconds": phases.get("generation", 0),
        "llm_call_counts": calls, "execution_diagnostics": {},
        "requested_answer_slot": "new advice", "query_time_scope": "mixed",
        "query_profile": {}, "base_query_profile": {}, "query_intent": {"intent": "new_advice"},
        "preference_profile": {"applicable": False, "constraints": [], "active_constraints": [], **skipped},
        "evidence_synthesis_result": {"applicable": False, **skipped},
        "ablations": [], "graph_anchor_coverage": {"applicable": False},
        "selected_evidence_anchor_coverage": {"applicable": False}, "actor_grounding": {"applicable": False},
        "candidate_extraction_diagnostics": {**skipped, "candidate_extraction_batch_count": 0,
            "selected_evidence_count": len(final), "evidence_absence_established": False},
        "operation_plan": operation, "inferred_operation_plan": operation,
        "operation_result": {"status": "not_applicable", "operation": "none"},
        "evidence_compiler_mode": args.pragmos_evidence_compiler_mode, "evidence_compiler_active": False,
        "label_blind_inference": True, "semantic_query_plan": None,
        "semantic_route_validation": {"eligible": False, **skipped},
        "evidence_compiler_state": "NOT_APPLICABLE", "evidence_compiler_ledger": [],
        "evidence_compiler_binding_diagnostics": skipped, "evidence_compiler_candidate_adjudication": skipped,
        "evidence_compiler_proof": skipped, "temporal_contract_result": {**skipped, "mode": "off", "calls": {}},
        "state_selector_plan": {"selector": "none"}, "state_history_result": skipped,
        "state_attribute_mismatch_result": skipped, "ordinal_list_result": skipped, "typed_answer_result": skipped,
        "turn_count": len(turns), "memory_count": len(context.vector_memory),
        "graph_node_count": context.G.number_of_nodes(), "graph_edge_count": context.G.number_of_edges(),
        "retrieval_candidate_pool": candidates, "final_memories": final,
        "evidence_synthesis_retrieval_diagnostics": {"applicable": False, **skipped},
        "temporal_date_index_diagnostics": {"applicable": False, **skipped},
    }
    for name in ("graph_materialization", "graph_retrieval", "temporal_date_index", "candidate_extraction",
                 "operation", "semantic_planner", "temporal_contract"):
        result[name + "_seconds"] = 0.0
    for name in ("temporal_fact_candidates", "answer_slot_candidates", "preliminary_memories", "session_diverse_memories",
                 "evidence_synthesis_memories", "operation_targeted_retrieval", "graph_neighbor_memories",
                 "graph_materialization_memories", "graph_extractions", "candidate_extractions",
                 "answer_candidate_extractions", "selected_candidate_memories", "operation_user_turn_memories",
                 "temporal_date_indexed_memories", "graph_evidence", "session_neighbor_memories"):
        result[name] = []
    return result, report
