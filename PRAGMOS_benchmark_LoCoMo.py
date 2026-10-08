#!/usr/bin/env python3

"""LoCoMo benchmark adapter for raw Phi-3 and PRAGMOS.

The adapter translates LoCoMo's conversation/session schema into the generic
structured-turn interface used by PRAGMOS. Gold answers and evidence IDs are
kept out of inference records and are used only after prediction for scoring.
"""

import argparse
import copy
import datetime
import json
import math
import re
import statistics
import string
import time
from collections import Counter
from pathlib import Path

from Phi3_raw_baseline import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_GPU_LAYERS,
    DEFAULT_LLM_PATH,
    DEFAULT_NUM_THREADS,
    DEFAULT_OFFLOAD_KQV,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    Phi3RawChat,
)
from PRAGMOS_benchmark_LongMemEval import (
    PRAGMOS_ABLATION_CHOICES,
    PRAGMOS_SYSTEM_PROMPT,
    atomic_write_json,
    build_raw_phi3_haystack_question,
    flush_and_sync,
    local_file_identity,
    pragmos_ablation_set,
    prepare_run_checkpoint,
    run_pragmos_record,
    with_manifest_fingerprint,
)


DEFAULT_DATA_FILE = (
    "benchmark/locomo/synthetic_v1/"
    "pragmos_synthetic_locomo_1986_v1.json"
)
DEFAULT_OUTPUT_DIR = "benchmark/locomo/runs"
DEFAULT_MAX_TOKENS = 64
RETRIEVAL_KS = (1, 5, 10)
MANIFEST_SCHEMA_VERSION = 1
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
CATEGORY_TO_PRAGMOS_QUESTION_TYPE = {
    1: "multi-session",
    2: "temporal-reasoning",
    3: "open-domain",
    4: "single-session-user",
}
LOCOMO_ANSWER_POLICY = (
    "Answer only from the supplied conversation evidence, except when the "
    "question explicitly requires ordinary background knowledge. Treat both "
    "named speakers as people whose statements may contain facts. Return only "
    "the concise answer, with no explanation, citation, or Answer label. If the "
    "evidence does not support the requested person and attribute, answer "
    "exactly: No information available."
)
LOCOMO_RAW_SYSTEM_PROMPT = (
    "You are answering a LoCoMo conversational-memory question using the "
    "provided prior conversation history. " + LOCOMO_ANSWER_POLICY
)
LOCOMO_PRAGMOS_SYSTEM_PROMPT = (
    PRAGMOS_SYSTEM_PROMPT
    + " The evidence is a conversation between named peers; do not treat either "
    "peer as the answering assistant. "
    + LOCOMO_ANSWER_POLICY
)
TEMPORAL_INSTRUCTION = (
    " Use the dates attached to the conversation sessions. When the evidence "
    "uses relative time, resolve it against that session date and return the "
    "corresponding approximate calendar date."
)


def load_locomo(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LoCoMo data file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("LoCoMo data must be a top-level JSON list")
    return data


def session_number(key):
    match = re.fullmatch(r"session_(\d+)", str(key))
    return int(match.group(1)) if match else None


def locomo_session_keys(conversation):
    keys = [
        key
        for key in conversation
        if session_number(key) is not None
        and isinstance(conversation.get(key), list)
    ]
    return sorted(keys, key=session_number)


def normalize_session_timestamp(value):
    """Convert released LoCoMo timestamps to ISO without inventing a timezone."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text
    for pattern in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%d %B, %Y",
        "%d %B %Y",
    ):
        try:
            parsed = datetime.datetime.strptime(text.upper(), pattern)
            return parsed.isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unsupported LoCoMo session timestamp: {value!r}")


def validate_locomo_dataset(samples):
    seen_question_ids = set()
    seen_sample_ids = set()
    category_counts = Counter()
    question_count = 0
    turn_count = 0
    caption_count = 0

    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"LoCoMo sample {sample_index} is not an object")
        sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate LoCoMo sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"Sample {sample_id} has no conversation object")
        session_keys = locomo_session_keys(conversation)
        if not session_keys:
            raise ValueError(f"Sample {sample_id} has no sessions")

        dialog_ids = set()
        for key in session_keys:
            normalize_session_timestamp(conversation.get(f"{key}_date_time"))
            for turn in conversation[key]:
                if not isinstance(turn, dict):
                    raise ValueError(f"{sample_id}/{key} contains a non-object turn")
                dialog_id = str(turn.get("dia_id") or "").strip()
                speaker = str(turn.get("speaker") or "").strip()
                text = str(turn.get("text") or "").strip()
                if not dialog_id or dialog_id in dialog_ids:
                    raise ValueError(
                        f"{sample_id}/{key} has a missing or duplicate dia_id"
                    )
                if not speaker or not text:
                    raise ValueError(
                        f"{sample_id}/{dialog_id} has an empty speaker or text"
                    )
                dialog_ids.add(dialog_id)
                turn_count += 1
                if str(turn.get("blip_caption") or "").strip():
                    caption_count += 1

        qas = sample.get("qa")
        if not isinstance(qas, list):
            raise ValueError(f"Sample {sample_id} has no qa list")
        for qa_index, qa in enumerate(qas):
            if not isinstance(qa, dict):
                raise ValueError(f"{sample_id} QA {qa_index} is not an object")
            question_id = str(
                qa.get("question_id") or f"{sample_id}:q{qa_index + 1:04d}"
            )
            if question_id in seen_question_ids:
                raise ValueError(f"Duplicate LoCoMo question_id: {question_id}")
            seen_question_ids.add(question_id)
            question = str(qa.get("question") or "").strip()
            category = qa.get("category")
            evidence = qa.get("evidence") or []
            if not question or category not in CATEGORY_NAMES:
                raise ValueError(f"Invalid question/category for {question_id}")
            if not isinstance(evidence, list) or not set(evidence).issubset(dialog_ids):
                raise ValueError(f"Invalid evidence IDs for {question_id}")
            answer = qa.get("answer")
            if category != 5 and (answer is None or answer == ""):
                raise ValueError(f"Answerable QA {question_id} has no answer")
            category_counts[int(category)] += 1
            question_count += 1

    return {
        "sample_count": len(samples),
        "question_count": question_count,
        "turn_count": turn_count,
        "caption_turn_count": caption_count,
        "category_counts": {
            str(category): category_counts[category]
            for category in sorted(CATEGORY_NAMES)
        },
    }


def build_locomo_turns(context_layer, sample):
    """Translate one LoCoMo conversation into peer-speaker structured turns."""
    conversation = sample["conversation"]
    turns = []
    internal_to_dialog = {}
    dialog_to_session = {}

    for session_key in locomo_session_keys(conversation):
        timestamp = normalize_session_timestamp(
            conversation.get(f"{session_key}_date_time")
        )
        for raw_turn in conversation[session_key]:
            speaker = str(raw_turn["speaker"]).strip()
            dialog_id = str(raw_turn["dia_id"]).strip()
            utterance = str(raw_turn["text"]).strip()
            turn = context_layer.create_turn(
                role="user",
                speaker=speaker,
                text=utterance,
                session_id=session_key,
                timestamp=timestamp,
                external_turn_id=dialog_id,
                evidence_type="dialogue",
            )
            turns.append(turn)
            internal_to_dialog[turn.turn_id] = dialog_id
            dialog_to_session[dialog_id] = session_key

            caption = str(raw_turn.get("blip_caption") or "").strip()
            if caption:
                caption_turn = context_layer.create_turn(
                    role="user",
                    speaker=speaker,
                    text=caption,
                    session_id=session_key,
                    timestamp=timestamp,
                    external_turn_id=f"{dialog_id}#caption",
                    evidence_type="image_caption",
                    parent_turn_id=dialog_id,
                )
                turns.append(caption_turn)
                internal_to_dialog[caption_turn.turn_id] = dialog_id

    return {
        "turns": turns,
        "internal_to_dialog": internal_to_dialog,
        "dialog_to_session": dialog_to_session,
    }


def locomo_as_haystack_record(sample):
    conversation = sample["conversation"]
    sessions = []
    session_ids = []
    session_dates = []
    for session_key in locomo_session_keys(conversation):
        messages = []
        for turn in conversation[session_key]:
            content = str(turn["text"]).strip()
            caption = str(turn.get("blip_caption") or "").strip()
            if caption:
                content += f"\n[Image caption: {caption}]"
            messages.append(
                {
                    "role": str(turn["speaker"]).strip(),
                    "content": content,
                }
            )
        sessions.append(messages)
        session_ids.append(session_key)
        session_dates.append(
            normalize_session_timestamp(
                conversation.get(f"{session_key}_date_time")
            )
        )
    return {
        "haystack_sessions": sessions,
        "haystack_session_ids": session_ids,
        "haystack_dates": session_dates,
    }


def inference_record_for_category(category):
    """Return only non-gold metadata needed by the inference pipeline."""
    record = {}
    question_type = CATEGORY_TO_PRAGMOS_QUESTION_TYPE.get(int(category))
    if question_type:
        record["question_type"] = question_type
    return record


def system_prompt_for_category(base_prompt, category):
    if int(category) == 2:
        return base_prompt + TEMPORAL_INSTRUCTION
    return base_prompt


def canonicalize_locomo_abstention(prediction):
    text = re.sub(r"\s+", " ", str(prediction or "")).strip()
    normalized = text.lower().strip(" .!?:;\"'")
    if (
        re.fullmatch(
            r"(?:i (?:do not|don't) know|unknown|not (?:known|provided|stated)|"
            r"insufficient (?:evidence|information)|no (?:relevant )?information "
            r"(?:is )?available|not mentioned)",
            normalized,
        )
        or "no information available" in normalized
        or "not mentioned" in normalized
    ):
        return "No information available"
    return text


def porter_stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as exc:
        raise ImportError(
            "LoCoMo's official-compatible local scorer requires NLTK. Install "
            "it in the active environment with `python -m pip install nltk`."
        ) from exc
    return PorterStemmer()


def normalize_locomo_answer(value):
    text = str(value or "").replace(",", "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def official_token_f1(prediction, reference, stemmer=None):
    stemmer = stemmer or porter_stemmer()
    prediction_tokens = [
        stemmer.stem(word) for word in normalize_locomo_answer(prediction).split()
    ]
    reference_tokens = [
        stemmer.stem(word) for word in normalize_locomo_answer(reference).split()
    ]
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def official_locomo_score(prediction, answer, category, stemmer=None):
    category = int(category)
    if category == 5:
        normalized = str(prediction or "").lower()
        return float(
            "no information available" in normalized
            or "not mentioned" in normalized
        )
    reference = str(answer or "")
    if category == 3:
        reference = reference.split(";", 1)[0].strip()
    if category in {2, 3, 4}:
        return official_token_f1(prediction, reference, stemmer=stemmer)
    if category == 1:
        predictions = [item.strip() for item in str(prediction).split(",")]
        references = [item.strip() for item in reference.split(",")]
        if not references:
            return 0.0
        return statistics.mean(
            max(
                official_token_f1(candidate, gold, stemmer=stemmer)
                for candidate in predictions
            )
            for gold in references
        )
    raise ValueError(f"Unsupported LoCoMo category: {category}")


def ranked_dialog_ids(items, internal_to_dialog):
    ranked = []
    seen = set()
    for item in items or []:
        dialog_id = item.get("parent_turn_id")
        if not dialog_id:
            external_id = item.get("external_turn_id")
            if external_id:
                dialog_id = str(external_id).split("#caption", 1)[0]
        if not dialog_id:
            source_turn_ids = item.get("source_turn_ids") or []
            for turn_id in source_turn_ids:
                if turn_id in internal_to_dialog:
                    dialog_id = internal_to_dialog[turn_id]
                    break
                if str(turn_id) in internal_to_dialog:
                    dialog_id = internal_to_dialog[str(turn_id)]
                    break
        if dialog_id and dialog_id not in seen:
            seen.add(dialog_id)
            ranked.append(str(dialog_id))
    return ranked


def ndcg_at_k(gold_ids, ranked_ids, k):
    gold = set(gold_ids)
    if not gold:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(ranked_ids[:k])
        if item in gold
    )
    ideal_count = min(len(gold), k)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(gold_ids, ranked_ids, ks=RETRIEVAL_KS):
    gold = list(dict.fromkeys(str(item) for item in gold_ids or []))
    ranked = list(dict.fromkeys(str(item) for item in ranked_ids or []))
    if not gold:
        return {
            "applicable": False,
            "gold_ids": [],
            "ranked_ids": ranked,
        }
    result = {
        "applicable": True,
        "gold_ids": gold,
        "ranked_ids": ranked,
        "gold_count": len(gold),
        "complete_evidence_coverage": set(gold).issubset(ranked),
    }
    for k in ks:
        result[f"recall_at_{k}"] = len(set(gold).intersection(ranked[:k])) / len(gold)
        result[f"ndcg_at_{k}"] = ndcg_at_k(gold, ranked, k)
    return result


def session_ids_for_dialogs(dialog_ids, dialog_to_session):
    return list(
        dict.fromkeys(
            dialog_to_session[dialog_id]
            for dialog_id in dialog_ids
            if dialog_id in dialog_to_session
        )
    )


def flatten_workload(samples, args):
    workload = []
    global_index = 0
    categories = set(args.category or [])
    sample_filter = set(args.sample_id or [])
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or f"sample-{sample_index}")
        for qa_index, qa in enumerate(sample.get("qa") or []):
            question_id = str(
                qa.get("question_id") or f"{sample_id}:q{qa_index + 1:04d}"
            )
            current_index = global_index
            global_index += 1
            if current_index < args.start_index:
                continue
            if sample_filter and sample_id not in sample_filter:
                continue
            if categories and int(qa["category"]) not in categories:
                continue
            workload.append(
                {
                    "dataset_index": current_index,
                    "question_id": question_id,
                    "question_id_value": question_id,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "sample": sample,
                    "qa": qa,
                }
            )
            if args.limit is not None and len(workload) >= args.limit:
                return workload
    return workload


def build_manifest(args, workload):
    ignored = {"flush_every", "output_dir", "resume", "run_name", "verbose_model"}
    config = {key: value for key, value in vars(args).items() if key not in ignored}
    data_identity = local_file_identity(args.data_file)
    model_identity = local_file_identity(args.model_path)
    config["data_file"] = data_identity["path"]
    config["model_path"] = model_identity["path"]
    root = Path(__file__).resolve().parent
    sources = [
        root / "PRAGMOS_benchmark_LoCoMo.py",
        root / "PRAGMOS_benchmark_LongMemEval.py",
        root / "Phi3_raw_baseline.py",
    ]
    if args.mode == "pragmos_context":
        sources.append(root / "PRAGMOS_context_layer_org.py")
    return with_manifest_fingerprint(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_name": args.run_name,
            "mode": args.mode,
            "data_source": data_identity["path"],
            "data_identity": {"kind": "local_file", **data_identity},
            "model_identity": model_identity,
            "source_identities": {
                path.name: local_file_identity(path) for path in sources
            },
            "config": config,
            "workload": [
                {
                    "dataset_index": item["dataset_index"],
                    "question_id": item["question_id"],
                    "sample_id": item["sample_id"],
                    "category": int(item["qa"]["category"]),
                }
                for item in workload
            ],
            "gold_isolation_policy": (
                "answer and evidence fields are read only after inference"
            ),
        }
    )


def aggregate_retrieval_metrics(rows, field):
    applicable = [row[field] for row in rows if row.get(field, {}).get("applicable")]
    if not applicable:
        return {"applicable_count": 0}
    result = {
        "applicable_count": len(applicable),
        "complete_evidence_coverage_rate": statistics.mean(
            float(item["complete_evidence_coverage"]) for item in applicable
        ),
    }
    for k in RETRIEVAL_KS:
        result[f"mean_recall_at_{k}"] = statistics.mean(
            item[f"recall_at_{k}"] for item in applicable
        )
        result[f"mean_ndcg_at_{k}"] = statistics.mean(
            item[f"ndcg_at_{k}"] for item in applicable
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run raw Phi-3 or PRAGMOS on LoCoMo-shaped data."
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument(
        "--mode",
        choices=["raw_phi3_haystack", "pragmos_context"],
        default="pragmos_context",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--category",
        action="append",
        type=int,
        choices=sorted(CATEGORY_NAMES),
        help="Filter a category; repeat to select more than one.",
    )
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--validate-only", action="store_true")

    parser.add_argument("--model-path", default=DEFAULT_LLM_PATH)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--n-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--n-gpu-layers", type=int, default=DEFAULT_GPU_LAYERS)
    parser.add_argument(
        "--no-offload-kqv",
        action="store_true",
        default=not DEFAULT_OFFLOAD_KQV,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repeat-penalty", type=float, default=DEFAULT_REPEAT_PENALTY)
    parser.add_argument("--haystack-order", choices=["recent-first", "chronological"], default="recent-first")
    parser.add_argument("--haystack-system-prompt", default=LOCOMO_RAW_SYSTEM_PROMPT)
    parser.add_argument("--pragmos-system-prompt", default=LOCOMO_PRAGMOS_SYSTEM_PROMPT)
    parser.add_argument("--no-question-date", action="store_true")
    parser.add_argument("--verbose-model", action="store_true")

    parser.add_argument("--pragmos-top-k", type=int, default=6)
    parser.add_argument("--pragmos-min-score", type=float, default=0.20)
    parser.add_argument("--pragmos-graph-candidates", type=int, default=4)
    parser.add_argument("--pragmos-retrieval-pool", type=int, default=24)
    parser.add_argument("--pragmos-multisession-graph-candidates", type=int, default=12)
    parser.add_argument("--pragmos-max-retrieval-sessions", type=int, default=12)
    parser.add_argument("--pragmos-graph-depth", type=int, default=2)
    parser.add_argument("--pragmos-graph-evidence", type=int, default=4)
    parser.add_argument("--pragmos-session-neighbor-radius", type=int, default=2)
    parser.add_argument("--pragmos-session-neighbors", type=int, default=4)
    parser.add_argument("--pragmos-answer-context-tokens", type=int, default=1300)
    parser.add_argument("--pragmos-chunk-words", type=int, default=160)
    parser.add_argument("--pragmos-chunk-overlap-words", type=int, default=32)
    parser.add_argument("--pragmos-embedding-batch-size", type=int, default=64)
    parser.add_argument("--pragmos-embedding-cache-size", type=int, default=20000)
    parser.add_argument("--pragmos-reranker-cache-size", type=int, default=20000)
    parser.add_argument(
        "--pragmos-ablate",
        action="append",
        choices=PRAGMOS_ABLATION_CHOICES,
        default=[],
    )

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="pragmos_locomo")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = load_locomo(args.data_file)
    validation = validate_locomo_dataset(samples)
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return

    workload = flatten_workload(samples, args)
    if not workload:
        raise ValueError("The selected LoCoMo workload is empty")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.run_name}_predictions.jsonl"
    trace_path = output_dir / f"{args.run_name}_trace.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    manifest_path = output_dir / f"{args.run_name}_manifest.json"
    official_eval_path = output_dir / f"{args.run_name}_official_eval.json"
    if not args.resume and official_eval_path.exists():
        raise FileExistsError(f"Run artifact already exists: {official_eval_path}")

    manifest = build_manifest(args, workload)
    checkpoint = prepare_run_checkpoint(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        trace_path=trace_path,
        summary_path=summary_path,
        manifest=manifest,
        workload=workload,
        resume=args.resume,
    )
    resumed_from_count = checkpoint["completed_count"]
    if checkpoint["repaired"]:
        print(f"[resume] Repaired checkpoint; continuing from {resumed_from_count}.")
    elif resumed_from_count:
        print(f"[resume] Continuing after {resumed_from_count} completed questions.")

    model = None
    context_layer = None
    if resumed_from_count < len(workload):
        if args.mode == "pragmos_context":
            from PRAGMOS_context_layer_org import ContextLayer

            context_layer = ContextLayer(
                session_id="locomo-bootstrap",
                model_path=args.model_path,
                context_length=args.n_ctx,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
                offload_kqv=not args.no_offload_kqv,
                seed=args.seed,
                verbose=args.verbose_model,
                enable_dense_retrieval="dense" not in pragmos_ablation_set(args),
                enable_lexical_retrieval="lexical" not in pragmos_ablation_set(args),
                enable_graph_retrieval="graph" not in pragmos_ablation_set(args),
                enable_reranker="reranker" not in pragmos_ablation_set(args),
                embedding_cache_size=args.pragmos_embedding_cache_size,
                reranker_cache_size=args.pragmos_reranker_cache_size,
            )
        else:
            model = Phi3RawChat(
                model_path=args.model_path,
                n_ctx=args.n_ctx,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
                offload_kqv=not args.no_offload_kqv,
                seed=args.seed,
                verbose=args.verbose_model,
            )

    predictions_handle = predictions_path.open("a", encoding="utf-8")
    trace_handle = trace_path.open("a", encoding="utf-8")
    run_rows = list(checkpoint["traces"])
    processed = resumed_from_count
    newly_processed = 0
    current_sample_id = None
    prepared = None
    raw_record = None
    conversation_ingestion_seconds = 0.0
    stemmer = porter_stemmer()
    started_at = time.perf_counter()

    try:
        for item in workload[resumed_from_count:]:
            qa = item["qa"]
            category = int(qa["category"])
            question = str(qa["question"])
            sample_id = item["sample_id"]
            if sample_id != current_sample_id:
                current_sample_id = sample_id
                if args.mode == "pragmos_context":
                    context_layer.reset_memory(session_id=f"locomo-{sample_id}")
                    prepared = build_locomo_turns(context_layer, item["sample"])
                    ingestion_started = time.perf_counter()
                    context_layer.index_raw_turns(
                        prepared["turns"],
                        chunk_words=args.pragmos_chunk_words,
                        overlap_words=args.pragmos_chunk_overlap_words,
                        batch_size=args.pragmos_embedding_batch_size,
                    )
                    conversation_ingestion_seconds = time.perf_counter() - ingestion_started
                else:
                    raw_record = locomo_as_haystack_record(item["sample"])

            pragmos_result = None
            haystack_info = None
            if args.mode == "pragmos_context":
                query_layer = context_layer.fork_for_query()
                query_args = copy.copy(args)
                query_args.pragmos_system_prompt = system_prompt_for_category(
                    args.pragmos_system_prompt,
                    category,
                )
                pragmos_result = run_pragmos_record(
                    context_layer=query_layer,
                    record=inference_record_for_category(category),
                    question=question,
                    args=query_args,
                    question_id=item["question_id"],
                    prepared_turns=prepared["turns"],
                    memory_preindexed=True,
                    benchmark_session_prefix="locomo",
                    question_session_prefix="locomo-question",
                )
                context_layer.local_reranker = query_layer.local_reranker
                context_layer.local_reranker_load_attempted = (
                    query_layer.local_reranker_load_attempted
                )
                result = pragmos_result
            else:
                system_prompt = system_prompt_for_category(
                    args.haystack_system_prompt,
                    category,
                )
                haystack_info = build_raw_phi3_haystack_question(
                    model=model,
                    record=raw_record,
                    question=question,
                    system_prompt=system_prompt,
                    n_ctx=args.n_ctx,
                    max_tokens=args.max_tokens,
                    include_question_date=False,
                    haystack_order=args.haystack_order,
                )
                result = model.chat(
                    user_input=haystack_info["user_input"],
                    system_prompt=system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                )

            prediction = canonicalize_locomo_abstention(result["text"])

            # Gold fields are deliberately first accessed after inference.
            answer = qa.get("answer")
            gold_evidence = [str(value) for value in qa.get("evidence") or []]
            official_score = official_locomo_score(
                prediction,
                answer,
                category,
                stemmer=stemmer,
            )
            dialog_retrieval = {"applicable": False, "gold_ids": [], "ranked_ids": []}
            session_retrieval = {"applicable": False, "gold_ids": [], "ranked_ids": []}
            ranked_dialogs = []
            if pragmos_result is not None and category != 5:
                ranked_dialogs = ranked_dialog_ids(
                    list(pragmos_result["final_memories"])
                    + list(pragmos_result["graph_evidence"]),
                    prepared["internal_to_dialog"],
                )
                dialog_retrieval = retrieval_metrics(gold_evidence, ranked_dialogs)
                gold_sessions = session_ids_for_dialogs(
                    gold_evidence,
                    prepared["dialog_to_session"],
                )
                ranked_sessions = session_ids_for_dialogs(
                    ranked_dialogs,
                    prepared["dialog_to_session"],
                )
                session_retrieval = retrieval_metrics(gold_sessions, ranked_sessions)

            prediction_record = {
                "question_id": item["question_id"],
                "hypothesis": prediction,
            }
            trace_record = {
                **prediction_record,
                "mode": args.mode,
                "dataset_index": item["dataset_index"],
                "sample_id": sample_id,
                "qa_index": item["qa_index"],
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "question": question,
                "answer": answer,
                "evidence": gold_evidence,
                "official_locomo_f1": official_score,
                "elapsed_seconds": result["elapsed_seconds"],
                "prompt_tokens_estimate": result["prompt_tokens_estimate"],
                "completion_tokens_estimate": result["completion_tokens_estimate"],
                "canonicalized_abstention": prediction != result["text"],
                "dialog_retrieval_metrics": dialog_retrieval,
                "session_retrieval_metrics": session_retrieval,
            }
            if haystack_info is not None:
                trace_record["included_haystack_session_ids"] = haystack_info[
                    "included_haystack_session_ids"
                ]
                trace_record["included_haystack_session_count"] = haystack_info[
                    "included_haystack_session_count"
                ]
                trace_record["history_tokens_estimate"] = haystack_info[
                    "history_tokens_estimate"
                ]
            if pragmos_result is not None:
                trace_record.update(
                    {
                        "conversation_ingestion_seconds": conversation_ingestion_seconds,
                        "ingested_turn_count": len(prepared["turns"]),
                        "answer_decision": pragmos_result["answer_decision"],
                        "safe_abstention": pragmos_result["safe_abstention"],
                        "abstention_reason": pragmos_result["abstention_reason"],
                        "retrieval_seconds": pragmos_result["retrieval_seconds"],
                        "generation_seconds": pragmos_result["generation_seconds"],
                        "context_tokens_estimate": pragmos_result[
                            "context_tokens_estimate"
                        ],
                        "operation_plan": pragmos_result["operation_plan"],
                        "operation_result": pragmos_result["operation_result"],
                        "evidence_synthesis_result": pragmos_result[
                            "evidence_synthesis_result"
                        ],
                        "evidence_synthesis_retrieval_diagnostics": pragmos_result[
                            "evidence_synthesis_retrieval_diagnostics"
                        ],
                        "evidence_synthesis_memories": pragmos_result[
                            "evidence_synthesis_memories"
                        ],
                        "collection_retrieval_diagnostics": pragmos_result[
                            "collection_retrieval_diagnostics"
                        ],
                        "collection_memories": pragmos_result[
                            "collection_memories"
                        ],
                        "temporal_chain_retrieval_diagnostics": pragmos_result[
                            "temporal_chain_retrieval_diagnostics"
                        ],
                        "temporal_chain_memories": pragmos_result[
                            "temporal_chain_memories"
                        ],
                        "query_profile": pragmos_result["query_profile"],
                        "query_intent": pragmos_result["query_intent"],
                        "actor_grounding": pragmos_result["actor_grounding"],
                        "retrieval_candidate_pool": pragmos_result[
                            "retrieval_candidate_pool"
                        ],
                        "candidate_extractions": pragmos_result[
                            "candidate_extractions"
                        ],
                        "answer_candidate_extractions": pragmos_result[
                            "answer_candidate_extractions"
                        ],
                        "graph_evidence": pragmos_result["graph_evidence"],
                        "final_memories": pragmos_result["final_memories"],
                        "ranked_dialog_ids": ranked_dialogs,
                    }
                )

            predictions_handle.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            trace_handle.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            run_rows.append(trace_record)
            processed += 1
            newly_processed += 1
            if newly_processed % max(1, args.flush_every) == 0:
                flush_and_sync(predictions_handle, trace_handle)
            print(
                f"[{processed}/{len(workload)}] {item['question_id']} "
                f"category={category} f1={official_score:.3f} "
                f"time={result['elapsed_seconds']:.2f}s"
            )
    finally:
        try:
            flush_and_sync(predictions_handle, trace_handle)
        finally:
            predictions_handle.close()
            trace_handle.close()

    category_summary = {}
    for category in sorted(CATEGORY_NAMES):
        rows = [row for row in run_rows if int(row["category"]) == category]
        if rows:
            category_summary[str(category)] = {
                "name": CATEGORY_NAMES[category],
                "count": len(rows),
                "mean_official_locomo_f1": statistics.mean(
                    row["official_locomo_f1"] for row in rows
                ),
            }
    summary = {
        "mode": args.mode,
        "processed": len(run_rows),
        "target_record_count": len(workload),
        "newly_processed": newly_processed,
        "resumed_from_count": resumed_from_count,
        "resume_enabled": args.resume,
        "checkpoint_repaired": checkpoint["repaired"],
        "dataset_validation": validation,
        "avg_official_locomo_f1": statistics.mean(
            row["official_locomo_f1"] for row in run_rows
        ),
        "by_category": category_summary,
        "avg_elapsed_seconds": statistics.mean(
            row["elapsed_seconds"] for row in run_rows
        ),
        "invocation_elapsed_seconds": time.perf_counter() - started_at,
        "predictions_path": str(predictions_path),
        "trace_path": str(trace_path),
        "official_eval_path": str(official_eval_path),
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": manifest["fingerprint"],
        "scoring_policy": (
            "Local implementation of the pinned LoCoMo category 1-5 QA scorer"
        ),
        "gold_isolation_policy": (
            "Gold answer/evidence accessed only after each prediction"
        ),
    }
    if args.mode == "pragmos_context":
        summary["dialog_retrieval"] = aggregate_retrieval_metrics(
            run_rows,
            "dialog_retrieval_metrics",
        )
        summary["session_retrieval"] = aggregate_retrieval_metrics(
            run_rows,
            "session_retrieval_metrics",
        )
    official_eval_rows = [
        {
            "question_id": row["question_id"],
            "category": row["category"],
            "answer": row["answer"],
            "prediction": row["hypothesis"],
            "evidence": row["evidence"],
            "prediction_context": row.get("ranked_dialog_ids", []),
        }
        for row in run_rows
    ]
    atomic_write_json(official_eval_path, official_eval_rows)
    atomic_write_json(summary_path, summary)
    print(f"\nOverall LoCoMo F1: {summary['avg_official_locomo_f1']:.4f}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote trace: {trace_path}")
    print(f"Wrote official-eval rows: {official_eval_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
