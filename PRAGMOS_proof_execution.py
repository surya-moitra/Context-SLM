"""Conservative scheduling guards; no question parsing or new reasoning rules."""

import copy
import hashlib
import json


def candidate_identity(question, sources, integrity, contract, *, n_ctx, max_tokens, semantic_plan=None):
    contract = {key: value for key, value in contract.items() if key != "baseline_answer"}
    payload = [question, [source.as_dict() for source in sources.values()], integrity,
               contract, n_ctx, max_tokens, semantic_plan]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reuse_candidate(context, cached, *, baseline_answer, max_answer_tokens):
    """Re-execute only the pure baseline-dependent proof, never model selection."""
    report = copy.deepcopy(cached["report"])
    prepared = cached.get("prepared") or {}
    if prepared and report.get("selected_engine") == "typed_v4":
        from PRAGMOS_typed_computation import execute_typed_plan
        from PRAGMOS_composable_execution import ContractError
        try:
            proof = execute_typed_plan(prepared["request"], prepared["claims"], prepared["windows"],
                                       prepared["sources"], baseline_answer=baseline_answer)
        except (ContractError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
            report.update(status="NOT_PROVEN", answer=None, supported_abstention=False, reason=str(exc))
            report["execution_reuse"] = "identical_sources_and_contract; pure_baseline_recheck_failed"
            return report
        report.update(proof=proof, status=proof["status"], answer=proof["answer"],
                      reason=proof["reason"], coverage=proof["coverage"],
                      supported_abstention=bool(proof["abstention_supported"]))
        if report["supported_abstention"]:
            report["answer"] = "I do not know."
        if report.get("answer") and context.count_tokens(report["answer"]) > max_answer_tokens:
            report.update(status="NOT_PROVEN", answer=None, supported_abstention=False,
                          reason="typed_answer_exceeds_output_budget")
    report["execution_reuse"] = "identical_sources_and_contract; pure_baseline_recheck"
    return report


def independent_positive_guard(report, plan, operation, turn_lookup, *, windows=None):
    """Keep the authoritative numeric-input veto on the fast path too."""
    from PRAGMOS_composable_candidate import promotable
    from PRAGMOS_evidence_compiler import (
        OperationProof, _authoritative_numeric_claim, validate_numeric_operation_proof_sources,
    )
    diagnostics = {"accepted": False, "reason": "no_independent_positive_proof"}
    turn_lookup = {str(key): value for key, value in turn_lookup.items()}
    if not promotable(report):
        return diagnostics
    proof = report["proof"]
    if proof.get("baseline_veto") or report.get("supported_abstention"):
        return diagnostics
    numeric_guard = operation in {"sum", "average", "difference", "argmax", "argmin"}
    facts = []
    for row in proof["provenance"]:
        source = row.get("source") or {}
        turn = turn_lookup.get(str(source.get("turn_id")))
        if turn is None:
            diagnostics["reason"] = "proof_source_unresolved"
            return diagnostics
        field = turn.get if isinstance(turn, dict) else lambda key: getattr(turn, key, None)
        if any(str(field(key)) != str(source.get(key)) for key in
               ("session_id", "role", "speaker", "timestamp", "text")):
            diagnostics["reason"] = "proof_source_changed"
            return diagnostics
        claim = row.get("claim") or row
        quantity = claim.get("quantity")
        if not quantity or quantity.get("canonical_value") is None:
            continue
        window = (windows or {}).get(claim.get("window_id"))
        quote = row.get("quote") or (window.text if window is not None else source["text"])
        roles = claim.get("roles") or {}
        fact = {"value": quantity["canonical_value"], "status": claim.get("status"),
                "source_turn_id": source["turn_id"], "source_session_id": source["session_id"],
                "source_quote": source["text"], "clause": quote,
                "subject": roles.get("subject") or claim.get("entity"),
                "object": roles.get("object"), "label": roles.get("label"),
                "instrument": roles.get("instrument"), "theme": roles.get("theme")}
        # An unresolved qualification is not permission to bypass the old veto.
        if numeric_guard and _authoritative_numeric_claim(fact, turn_lookup) is None:
            diagnostics["reason"] = "numeric_source_qualification_unresolved"
            return diagnostics
        facts.append(fact)
    if numeric_guard and not facts:
        diagnostics["reason"] = "numeric_proof_has_no_bound_measurements"
        return diagnostics
    _, validation = validate_numeric_operation_proof_sources(
        plan, OperationProof("COMPLETE", plan.operator, report["answer"],
                             calculation={"provenance": facts}, executable=True),
        turn_lookup, effective_operation=operation,
    )
    diagnostics["source_input_validation"] = validation
    diagnostics.update(accepted=not validation["blocked"],
                       reason="positive_proof_sources_validated" if not validation["blocked"]
                       else "authoritative_numeric_inputs_incomplete")
    return diagnostics
