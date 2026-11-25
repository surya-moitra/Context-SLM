"""Execution-only controls shared by the benchmark adapters."""

import time
from concurrent.futures import ThreadPoolExecutor


def add_execution_arguments(parser):
    parser.add_argument(
        "--pragmos-factual-mode", choices=["off", "guarded"], default="off",
        help="Opt-in source-bound property reads and positive-proof scope admission; no gold labels are used.",
    )
    parser.add_argument(
        "--pragmos-assistance-mode", choices=["off", "guarded"], default="off",
        help="Opt-in new-advice route with question/source interpretation. Off preserves existing inference.",
    )
    parser.add_argument(
        "--pragmos-assistance-boundary", choices=["off", "observe"], default="off",
        help="Question-only, state-isolated purpose observation; cannot change evidence or answers.",
    )
    parser.add_argument(
        "--pragmos-execution-policy", choices=["reference", "optimized"],
        default="reference",
        help="Optimized batches independent retrieval work; reasoning and evidence limits are unchanged.",
    )
    parser.add_argument(
        "--pragmos-retrieval-device", choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for both local retrieval models, not the answer LLM.",
    )
    parser.add_argument(
        "--pragmos-overlap-planner", action="store_true",
        help="Overlap the initial LLM planner with CPU indexing; requires optimized policy and CPU retrieval.",
    )
    parser.add_argument(
        "--pragmos-proof-first", action="store_true",
        help="Try isolated, source-backed positive proofs before duplicate legacy extraction; otherwise fall back.",
    )


def validate_execution_arguments(args):
    factual = getattr(args, "pragmos_factual_mode", "off")
    if factual not in {"off", "guarded"}:
        raise ValueError("--pragmos-factual-mode supports only off or guarded")
    if factual == "guarded" and (args.mode != "pragmos_context" or getattr(args, "pragmos_ablate", None)):
        raise ValueError("guarded factual reads require pragmos_context without ablations")
    assistance = getattr(args, "pragmos_assistance_mode", "off")
    if assistance not in {"off", "guarded"}:
        raise ValueError("--pragmos-assistance-mode supports only off or guarded")
    if assistance == "guarded":
        if args.mode != "pragmos_context":
            raise ValueError("guarded assistance requires --mode pragmos_context")
        if getattr(args, "pragmos_assistance_boundary", "off") != "off":
            raise ValueError("guarded assistance cannot be combined with the frozen observation probe")
        if getattr(args, "pragmos_ablate", None):
            raise ValueError("guarded assistance ablations are not yet supported")
        if (getattr(args, "temperature", 0) != 0 or getattr(args, "top_p", 1) != 1
                or getattr(args, "repeat_penalty", 1.1) != 1.1):
            raise ValueError("guarded assistance currently requires temperature=0, top_p=1, repeat_penalty=1.1")
    boundary = getattr(args, "pragmos_assistance_boundary", "off")
    if boundary not in {"off", "observe"}:
        raise ValueError("--pragmos-assistance-boundary supports only off or observe")
    if boundary == "observe" and args.mode != "pragmos_context":
        raise ValueError("--pragmos-assistance-boundary observe requires --mode pragmos_context")
    if getattr(args, "pragmos_proof_first", False) and (
        args.mode != "pragmos_context"
        or getattr(args, "pragmos_composable_mode", "off") != "candidate"
        or getattr(args, "pragmos_composable_protocol", "slots_v3") != "typed_v4"
        or getattr(args, "pragmos_evidence_compiler_mode", "off") != "validated"
        or getattr(args, "pragmos_temporal_contract_mode", "off") != "off"
        or getattr(args, "pragmos_state_mode", "off") != "off"
        or getattr(args, "pragmos_overlap_planner", False)
    ):
        raise ValueError(
            "--pragmos-proof-first requires pragmos_context, candidate typed_v4, "
            "validated evidence compiler, temporal/state experiments off and no planner overlap"
        )
    if getattr(args, "pragmos_overlap_planner", False) and (
        getattr(args, "pragmos_execution_policy", "reference") != "optimized"
        or getattr(args, "pragmos_retrieval_device", "auto") != "cpu"
        or args.mode != "pragmos_context"
    ):
        raise ValueError(
            "--pragmos-overlap-planner requires --mode pragmos_context, "
            "--pragmos-execution-policy optimized and --pragmos-retrieval-device cpu"
        )


def index_with_planner(index, planner, *, overlap=False):
    """Run one planner and one independent indexer; never parallelize LLM calls.

    The caller must ensure the indexer uses CPU retrieval models and does not
    invoke the LLM. Workers are joined even if either operation fails.
    """
    started = time.perf_counter()

    def timed(operation):
        begin = time.perf_counter()
        return operation(), time.perf_counter() - begin

    if overlap and planner is not None:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pragmos-planner") as pool:
            pending = pool.submit(timed, planner)
            indexed, index_seconds = timed(index)
            planned, planner_seconds = pending.result()
    else:
        indexed, index_seconds = timed(index)
        planned, planner_seconds = timed(planner) if planner is not None else (None, 0.0)
    wall = time.perf_counter() - started
    return indexed, planned, {
        "index_seconds": index_seconds,
        "planner_seconds": planner_seconds,
        "index_and_planner_wall_seconds": wall,
        "overlap_enabled": bool(overlap and planner is not None),
        "overlapped_seconds": max(0.0, index_seconds + planner_seconds - wall),
        "timing_policy": "component_durations_overlap; use elapsed_seconds_for_end_to_end_latency",
    }


def retrieve_independent_queries(context, queries, **kwargs):
    """Batch only queries whose text and evidence index are already fixed."""
    queries = list(queries)
    if (getattr(context, "execution_policy", "reference") == "optimized"
            and callable(getattr(context, "retrieve_relevant_memories_many", None))):
        return context.retrieve_relevant_memories_many(queries, **kwargs)
    return [context.retrieve_relevant_memories(query, **kwargs) for query in queries]
