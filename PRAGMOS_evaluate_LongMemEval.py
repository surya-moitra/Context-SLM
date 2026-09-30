#!/usr/bin/env python3

"""Resumable wrapper around LongMemEval's official GPT-4o QA evaluator.

The wrapper imports the evaluator from a pinned LongMemEval checkout and uses
its prompt builder, model mapping, retry helper, request settings, and label
rule unchanged. It adds input validation, immutable run manifests, durable
JSONL checkpoints, resume support, and official metric aggregation.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1
OFFICIAL_METRIC_MODEL_SHORT = "gpt-4o"
OFFICIAL_METRIC_MODEL = "gpt-4o-2024-08-06"
OFFICIAL_QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_json_or_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}"
                    ) from exc
                records.append(value)
        return records
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def run_git(repo, *arguments):
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_official_evaluator(official_repo, expected_commit):
    official_repo = Path(official_repo).expanduser().resolve()
    if not (official_repo / ".git").exists():
        raise FileNotFoundError(
            f"Official evaluator is not a Git checkout: {official_repo}"
        )
    actual_commit = run_git(official_repo, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            "Official evaluator commit mismatch: expected "
            f"{expected_commit}, found {actual_commit}."
        )
    dirty_paths = run_git(official_repo, "status", "--porcelain")
    if dirty_paths:
        raise ValueError(
            "Official evaluator checkout has local changes. Restore a clean "
            "checkout before judging."
        )

    evaluator_path = official_repo / "src" / "evaluation" / "evaluate_qa.py"
    identity = file_identity(evaluator_path)
    spec = importlib.util.spec_from_file_location(
        "longmemeval_official_evaluate_qa",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if OFFICIAL_METRIC_MODEL_SHORT not in module.model_zoo:
        raise ValueError("Official evaluator does not define the gpt-4o judge.")
    metric_model, metric_source = module.model_zoo[OFFICIAL_METRIC_MODEL_SHORT]
    if metric_model != OFFICIAL_METRIC_MODEL or metric_source != "openai":
        raise ValueError(
            "Official evaluator model mapping changed; expected "
            f"{OFFICIAL_METRIC_MODEL_SHORT} -> {OFFICIAL_METRIC_MODEL}."
        )
    return module, {
        "repository_path": str(official_repo),
        "commit": actual_commit,
        "evaluator_file": identity,
    }


def create_official_client(official, api_key):
    import httpx

    return official.OpenAI(
        api_key=api_key,
        base_url=None,
        http_client=httpx.Client(),
    )


def validate_workload(predictions, references, start_index=0, limit=None):
    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    reference_by_id = {}
    for reference in references:
        if not isinstance(reference, dict) or "question_id" not in reference:
            raise ValueError("Every oracle row must contain question_id.")
        question_id = str(reference["question_id"])
        if question_id in reference_by_id:
            raise ValueError(f"Duplicate oracle question_id: {question_id}")
        for field in ("question", "answer", "question_type"):
            if field not in reference:
                raise ValueError(
                    f"Oracle question {question_id} is missing {field}."
                )
        reference_by_id[question_id] = reference

    seen_prediction_ids = set()
    validated = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Every prediction row must be a JSON object.")
        if "question_id" not in prediction or "hypothesis" not in prediction:
            raise ValueError(
                "Every prediction row must contain question_id and hypothesis."
            )
        question_id = str(prediction["question_id"])
        if question_id in seen_prediction_ids:
            raise ValueError(f"Duplicate prediction question_id: {question_id}")
        seen_prediction_ids.add(question_id)
        if question_id not in reference_by_id:
            raise ValueError(
                f"Prediction question_id is absent from oracle: {question_id}"
            )
        if not str(prediction["hypothesis"]).strip():
            raise ValueError(f"Prediction hypothesis is blank: {question_id}")
        validated.append(
            {
                "prediction": prediction,
                "reference": reference_by_id[question_id],
                "question_id": question_id,
            }
        )

    selected = validated[start_index:]
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        selected = selected[:limit]
    return selected


def fingerprint_manifest(manifest):
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "fingerprint"}
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    result = dict(manifest)
    result["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return result


def build_manifest(args, workload, evaluator_identity):
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_name": args.run_name,
        "metric_model_short": OFFICIAL_METRIC_MODEL_SHORT,
        "metric_model": OFFICIAL_METRIC_MODEL,
        "request_settings": {
            "n": 1,
            "temperature": 0,
            "max_tokens": 10,
        },
        "label_rule": "'yes' in eval_response.lower()",
        "predictions": file_identity(args.predictions),
        "oracle": file_identity(args.oracle),
        "official_evaluator": evaluator_identity,
        "wrapper": file_identity(Path(__file__).resolve()),
        "selection": {
            "start_index": args.start_index,
            "limit": args.limit,
            "expected_count": args.expected_count,
        },
        "workload": [
            {
                "question_id": row["question_id"],
                "hypothesis": row["prediction"]["hypothesis"],
            }
            for row in workload
        ],
    }
    return fingerprint_manifest(manifest)


def read_result_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return {"records": [], "offsets": [0], "file_size": 0}
    records = []
    offsets = [0]
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                stripped = line.strip()
                if not stripped:
                    raise ValueError("blank JSONL line")
                record = json.loads(stripped.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("result is not a JSON object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                if handle.read().strip():
                    raise ValueError(
                        f"Corrupt judge checkpoint before the final line: {path}"
                    ) from exc
                break
            records.append(record)
            offsets.append(handle.tell())
    return {
        "records": records,
        "offsets": offsets,
        "file_size": path.stat().st_size,
    }


def reconcile_results(path, workload):
    checkpoint = read_result_checkpoint(path)
    records = checkpoint["records"]
    if len(records) > len(workload):
        raise ValueError("Judge checkpoint is longer than the selected workload.")
    for index, result in enumerate(records):
        expected = workload[index]["prediction"]
        question_id = str(result.get("question_id"))
        if question_id != str(expected["question_id"]):
            raise ValueError(
                "Judge checkpoint is not a workload prefix at line "
                f"{index + 1}."
            )
        if result.get("hypothesis") != expected.get("hypothesis"):
            raise ValueError(
                "Judge checkpoint hypothesis mismatch at line "
                f"{index + 1} for {question_id}."
            )
        autoeval = result.get("autoeval_label") or {}
        if autoeval.get("model") != OFFICIAL_METRIC_MODEL:
            raise ValueError(
                f"Unexpected judge model at line {index + 1}: "
                f"{autoeval.get('model')!r}."
            )
        if not isinstance(autoeval.get("label"), bool):
            raise ValueError(f"Missing boolean judge label at line {index + 1}.")

    target_size = checkpoint["offsets"][len(records)]
    repaired = checkpoint["file_size"] != target_size
    if repaired:
        with Path(path).open("r+b") as handle:
            handle.truncate(target_size)
            handle.flush()
            os.fsync(handle.fileno())
    return records, repaired


def prepare_checkpoint(manifest_path, result_path, summary_path, manifest, workload, resume):
    manifest_path = Path(manifest_path)
    result_path = Path(result_path)
    summary_path = Path(summary_path)
    if not resume:
        existing = [
            str(path)
            for path in (manifest_path, result_path, summary_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Judge artifacts already exist. Use a new --run-name or the "
                "identical command with --resume. Existing: " + ", ".join(existing)
            )
        atomic_write_json(manifest_path, manifest)
        return [], False

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        verified = fingerprint_manifest(existing_manifest)
        if existing_manifest.get("fingerprint") != verified.get("fingerprint"):
            raise ValueError("Existing judge manifest is corrupt or was edited.")
        if existing_manifest.get("fingerprint") != manifest.get("fingerprint"):
            raise ValueError(
                "Judge configuration, inputs, wrapper, or official evaluator "
                "changed. Use a new --run-name instead of resuming."
            )
    else:
        legacy = [
            str(path) for path in (result_path, summary_path) if path.exists()
        ]
        if legacy:
            raise ValueError(
                "Cannot safely resume judge outputs without a manifest: "
                + ", ".join(legacy)
            )
        atomic_write_json(manifest_path, manifest)
    return reconcile_results(result_path, workload)


def official_metrics(results, reference_by_id):
    by_type = {question_type: [] for question_type in OFFICIAL_QUESTION_TYPES}
    abstention = []
    for result in results:
        question_id = str(result["question_id"])
        label = bool(result["autoeval_label"]["label"])
        question_type = reference_by_id[question_id]["question_type"]
        if question_type not in by_type:
            raise ValueError(f"Unsupported question_type: {question_type}")
        by_type[question_type].append(label)
        if "_abs" in question_id:
            abstention.append(label)

    type_metrics = {
        question_type: {
            "accuracy": (
                sum(labels) / len(labels) if labels else None
            ),
            "count": len(labels),
        }
        for question_type, labels in by_type.items()
    }
    all_labels = [label for labels in by_type.values() for label in labels]
    populated_type_accuracies = [
        metric["accuracy"]
        for metric in type_metrics.values()
        if metric["accuracy"] is not None
    ]
    return {
        "overall_accuracy": (
            sum(all_labels) / len(all_labels) if all_labels else None
        ),
        "task_averaged_accuracy": (
            sum(populated_type_accuracies) / len(populated_type_accuracies)
            if populated_type_accuracies
            else None
        ),
        "abstention_accuracy": (
            sum(abstention) / len(abstention) if abstention else None
        ),
        "abstention_count": len(abstention),
        "by_question_type": type_metrics,
    }


def write_summary(
    path,
    *,
    args,
    manifest,
    results,
    workload,
    reference_by_id,
    resumed_from_count,
    repaired,
    complete,
):
    metrics = official_metrics(results, reference_by_id)
    total_judge_seconds = sum(
        float((row.get("judge_audit") or {}).get("elapsed_seconds", 0.0))
        for row in results
    )
    summary = {
        "run_name": args.run_name,
        "metric_model_short": OFFICIAL_METRIC_MODEL_SHORT,
        "metric_model": OFFICIAL_METRIC_MODEL,
        "processed": len(results),
        "target_record_count": len(workload),
        "resumed_from_count": resumed_from_count,
        "resume_enabled": args.resume,
        "checkpoint_repaired": repaired,
        "complete": complete,
        "manifest_fingerprint": manifest["fingerprint"],
        "results_path": str(Path(args.output_dir) / f"{args.run_name}_results.jsonl"),
        "total_judge_seconds": total_judge_seconds,
        "avg_judge_seconds": (
            total_judge_seconds / len(results) if results else 0.0
        ),
        **metrics,
    }
    atomic_write_json(path, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe wrapper for LongMemEval's official GPT-4o QA judge."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--expected-official-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and official evaluator without making API calls.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = load_json_or_jsonl(args.predictions)
    references = load_json_or_jsonl(args.oracle)
    workload = validate_workload(
        predictions,
        references,
        start_index=args.start_index,
        limit=args.limit,
    )
    if args.expected_count is not None and len(workload) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} selected predictions, found "
            f"{len(workload)}."
        )

    official, evaluator_identity = load_official_evaluator(
        args.official_repo,
        args.expected_official_commit,
    )
    manifest = build_manifest(args, workload, evaluator_identity)
    if args.validate_only:
        client_probe = create_official_client(official, "validation-only")
        client_probe.close()
        print(
            json.dumps(
                {
                    "status": "valid",
                    "selected_predictions": len(workload),
                    "metric_model": OFFICIAL_METRIC_MODEL,
                    "official_commit": evaluator_identity["commit"],
                    "official_evaluator_sha256": evaluator_identity[
                        "evaluator_file"
                    ]["sha256"],
                    "predictions_sha256": manifest["predictions"]["sha256"],
                    "oracle_sha256": manifest["oracle"]["sha256"],
                    "manifest_fingerprint": manifest["fingerprint"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.run_name}_manifest.json"
    result_path = output_dir / f"{args.run_name}_results.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"
    results, repaired = prepare_checkpoint(
        manifest_path,
        result_path,
        summary_path,
        manifest,
        workload,
        args.resume,
    )
    resumed_from_count = len(results)
    if repaired:
        print(f"[resume] Repaired partial final result; continuing at {resumed_from_count}.")
    elif resumed_from_count:
        print(f"[resume] Continuing after {resumed_from_count} judged predictions.")

    reference_by_id = {
        str(reference["question_id"]): reference for reference in references
    }
    client = None
    if resumed_from_count < len(workload):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Create an API Platform key and "
                "export it only in the current shell."
            )
        official.openai.organization = os.getenv("OPENAI_ORGANIZATION")
        client = create_official_client(official, api_key)
    try:
        with result_path.open("a", encoding="utf-8") as output:
            newly_processed = 0
            for index, row in enumerate(
                workload[resumed_from_count:],
                start=resumed_from_count + 1,
            ):
                prediction = row["prediction"]
                reference = row["reference"]
                question_id = row["question_id"]
                prompt = official.get_anscheck_prompt(
                    reference["question_type"],
                    reference["question"],
                    reference["answer"],
                    prediction["hypothesis"],
                    abstention="_abs" in question_id,
                )
                started_at = time.perf_counter()
                completion = official.chat_completions_with_backoff(
                    client,
                    model=OFFICIAL_METRIC_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=0,
                    max_tokens=10,
                )
                elapsed_seconds = time.perf_counter() - started_at
                eval_response = completion.choices[0].message.content.strip()
                label = "yes" in eval_response.lower()
                result = dict(prediction)
                result["autoeval_label"] = {
                    "model": OFFICIAL_METRIC_MODEL,
                    "label": label,
                }
                result["judge_audit"] = {
                    "response": eval_response,
                    "elapsed_seconds": elapsed_seconds,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
                newly_processed += 1
                if newly_processed % max(1, args.flush_every) == 0:
                    output.flush()
                    os.fsync(output.fileno())
                print(
                    f"[{index}/{len(workload)}] {question_id} "
                    f"label={'yes' if label else 'no'} "
                    f"time={elapsed_seconds:.2f}s",
                    flush=True,
                )
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        write_summary(
            summary_path,
            args=args,
            manifest=manifest,
            results=results,
            workload=workload,
            reference_by_id=reference_by_id,
            resumed_from_count=resumed_from_count,
            repaired=repaired,
            complete=False,
        )
        raise
    finally:
        if client is not None:
            client.close()

    summary = write_summary(
        summary_path,
        args=args,
        manifest=manifest,
        results=results,
        workload=workload,
        reference_by_id=reference_by_id,
        resumed_from_count=resumed_from_count,
        repaired=repaired,
        complete=len(results) == len(workload),
    )
    print(f"\nOverall Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Task-averaged Accuracy: {summary['task_averaged_accuracy']:.4f}")
    if summary["abstention_accuracy"] is not None:
        print(f"Abstention Accuracy: {summary['abstention_accuracy']:.4f}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote results: {result_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
