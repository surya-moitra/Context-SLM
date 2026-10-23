#!/usr/bin/env python3
"""Run a benchmark unchanged while collecting publication-oriented resource data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI guard
    psutil = None


SCHEMA_VERSION = 1
DEFAULT_CORE_REF = "pragmos-lme-locomo-official-candidate-v1"
FROZEN_CORE_FILES = (
    "PRAGMOS_context_layer_org.py",
    "PRAGMOS_benchmark_LongMemEval.py",
    "Phi3_raw_baseline.py",
)
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CLOUD_CREDENTIAL_VARIABLES = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_git(repo_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_text(repo_root: Path, *arguments: str) -> str:
    return run_git(repo_root, *arguments).stdout.decode("utf-8").strip()


def verify_core_files(
    repo_root: Path,
    expected_ref: str,
    core_files: Sequence[str] = FROZEN_CORE_FILES,
) -> dict[str, Any]:
    """Verify that the current core bytes equal the frozen publication ref."""
    expected_commit = git_text(repo_root, "rev-parse", f"{expected_ref}^{{commit}}")
    identities = {}
    mismatches = []
    for relative_path in core_files:
        current_path = repo_root / relative_path
        if not current_path.is_file():
            raise FileNotFoundError(f"Frozen core file is missing: {current_path}")
        expected = run_git(repo_root, "show", f"{expected_ref}:{relative_path}").stdout
        current = current_path.read_bytes()
        expected_hash = sha256_bytes(expected)
        current_hash = sha256_bytes(current)
        matches = current_hash == expected_hash
        identities[relative_path] = {
            "current_sha256": current_hash,
            "expected_sha256": expected_hash,
            "matches_expected_ref": matches,
        }
        if not matches:
            mismatches.append(relative_path)
    if mismatches:
        names = ", ".join(mismatches)
        raise RuntimeError(
            f"Frozen PRAGMOS core differs from {expected_ref}: {names}. "
            "Use the validated core or explicitly choose the correct frozen ref."
        )
    return {
        "enabled": True,
        "expected_ref": expected_ref,
        "expected_commit": expected_commit,
        "files": identities,
    }


def tracked_worktree_dirty(repo_root: Path) -> bool:
    unstaged = run_git(repo_root, "diff", "--quiet", check=False).returncode != 0
    staged = run_git(repo_root, "diff", "--cached", "--quiet", check=False).returncode != 0
    return unstaged or staged


def command_option(command: Sequence[str], name: str) -> str | None:
    for index, argument in enumerate(command):
        if argument == name and index + 1 < len(command):
            return command[index + 1]
        prefix = f"{name}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def local_file_metadata(value: str | None, working_directory: Path) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = working_directory / path
    resolved = path.resolve()
    metadata: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.is_file():
        metadata["size_bytes"] = resolved.stat().st_size
    return metadata


def prepare_child_environment(
    environment: dict[str, str],
    *,
    local_only: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    child_environment = dict(environment)
    removed = []
    if local_only:
        for name in CLOUD_CREDENTIAL_VARIABLES:
            if name in child_environment:
                removed.append(name)
                child_environment.pop(name, None)
        child_environment.update(OFFLINE_ENVIRONMENT)
    return child_environment, {
        "local_only_requested": local_only,
        "cloud_credential_variables_removed": sorted(removed),
        "offline_environment": dict(OFFLINE_ENVIRONMENT) if local_only else {},
        "note": (
            "Local-only mode removes common hosted-service credentials and forces "
            "supported model hubs into offline mode; it is an execution control, "
            "not an operating-system network firewall."
            if local_only
            else "Local-only execution controls were not requested."
        ),
    }


def sysctl_value(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    result = subprocess.run(
        ["sysctl", "-n", name],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    value = result.stdout.strip()
    return value or None


def hardware_metadata() -> dict[str, Any]:
    if psutil is None:
        raise RuntimeError(
            "psutil is required. Install it with: "
            "python -m pip install -r requirements-measurement.txt"
        )
    physical_memory_bytes = int(psutil.virtual_memory().total)
    return {
        "hardware_model": sysctl_value("hw.model") or platform.node(),
        "cpu_model": sysctl_value("machdep.cpu.brand_string") or platform.processor(),
        "machine_architecture": platform.machine(),
        "operating_system": platform.platform(),
        "physical_memory_bytes": physical_memory_bytes,
        "physical_memory_gib": physical_memory_bytes / (1024**3),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True) or 1,
    }


def process_tree_sample(pid: int, elapsed_seconds: float) -> dict[str, Any]:
    if psutil is None:
        raise RuntimeError("psutil is required for resource sampling")
    tree_enumeration_complete = True
    try:
        root = psutil.Process(pid)
        try:
            processes = [root, *root.children(recursive=True)]
        except (psutil.AccessDenied, PermissionError, OSError):
            processes = [root]
            tree_enumeration_complete = False
    except psutil.NoSuchProcess:
        processes = []

    rss_bytes = 0
    sampled_pids = set()
    for process in processes:
        if process.pid in sampled_pids:
            continue
        try:
            rss_bytes += int(process.memory_info().rss)
            sampled_pids.add(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
            continue
    return {
        "elapsed_seconds": elapsed_seconds,
        "process_count": len(sampled_pids),
        "rss_bytes": rss_bytes,
        "tree_enumeration_complete": tree_enumeration_complete,
    }


def cpu_seconds_for_children() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def resolve_processed_questions(
    benchmark_summary_path: Path | None,
    explicit_question_count: int | None,
) -> tuple[int | None, str | None]:
    if benchmark_summary_path is not None and benchmark_summary_path.is_file():
        summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
        if "newly_processed" in summary:
            return int(summary["newly_processed"]), "benchmark_summary.newly_processed"
        if "processed" in summary:
            return int(summary["processed"]), "benchmark_summary.processed"
    if explicit_question_count is not None:
        return explicit_question_count, "command_line.question_count"
    return None, None


def calculate_headline_metrics(
    *,
    peak_rss_bytes: int,
    cpu_seconds: float,
    wall_seconds: float,
    logical_cpu_count: int,
    processed_questions: int | None,
) -> dict[str, float | None]:
    mean_cpu = None
    if wall_seconds > 0 and logical_cpu_count > 0:
        mean_cpu = cpu_seconds / wall_seconds / logical_cpu_count * 100.0
    cpu_per_question = None
    if processed_questions is not None and processed_questions > 0:
        cpu_per_question = cpu_seconds / processed_questions
    return {
        "peak_process_tree_rss_gib": peak_rss_bytes / (1024**3),
        "mean_cpu_utilization_percent_of_machine": mean_cpu,
        "cpu_seconds_per_processed_question": cpu_per_question,
    }


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def validate_output_paths(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Measurement artifacts already exist: {names}. "
            "Use a new run name or pass --overwrite."
        )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-summary", type=Path)
    parser.add_argument("--question-count", type=int)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--expected-core-ref", default=DEFAULT_CORE_REF)
    parser.add_argument("--skip-core-check", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a benchmark command is required after --")
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name may contain only letters, numbers, '.', '_' and '-'")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    if args.question_count is not None and args.question_count < 0:
        parser.error("--question-count must be non-negative")
    return args


def run_measurement(args: argparse.Namespace) -> int:
    if psutil is None:
        raise RuntimeError(
            "psutil is required. Install it with: "
            "python -m pip install -r requirements-measurement.txt"
        )

    working_directory = args.working_directory.expanduser().resolve()
    repo_root = Path(git_text(working_directory, "rev-parse", "--show-toplevel"))
    output_directory = args.output_dir.expanduser()
    if not output_directory.is_absolute():
        output_directory = working_directory / output_directory
    output_directory = output_directory.resolve()
    benchmark_summary_path = args.benchmark_summary
    if benchmark_summary_path is not None:
        benchmark_summary_path = benchmark_summary_path.expanduser()
        if not benchmark_summary_path.is_absolute():
            benchmark_summary_path = working_directory / benchmark_summary_path
        benchmark_summary_path = benchmark_summary_path.resolve()

    manifest_path = output_directory / f"{args.run_name}_measurement_manifest.json"
    samples_path = output_directory / f"{args.run_name}_resource_samples.jsonl"
    summary_path = output_directory / f"{args.run_name}_resource_summary.json"
    validate_output_paths(
        (manifest_path, samples_path, summary_path),
        overwrite=args.overwrite,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.skip_core_check:
        core_guard = {
            "enabled": False,
            "expected_ref": args.expected_core_ref,
            "note": "Core verification was explicitly skipped; do not use this mode for publication runs.",
        }
    else:
        core_guard = verify_core_files(repo_root, args.expected_core_ref)

    hardware = hardware_metadata()
    child_environment, local_execution = prepare_child_environment(
        dict(os.environ),
        local_only=args.local_only,
    )
    model_metadata = local_file_metadata(
        command_option(args.command, "--model-path"),
        working_directory,
    )
    data_metadata = local_file_metadata(
        command_option(args.command, "--data-file"),
        working_directory,
    )
    if args.local_only:
        for label, metadata in (("model", model_metadata), ("data", data_metadata)):
            if metadata is not None and not metadata["exists"]:
                raise FileNotFoundError(
                    f"Local-only {label} path does not exist: {metadata['path']}"
                )
        remote_arguments = [
            value for value in args.command if value.startswith(("http://", "https://"))
        ]
        if remote_arguments:
            raise ValueError(
                "Local-only command contains remote URL arguments: "
                + ", ".join(remote_arguments)
            )

    current_commit = git_text(repo_root, "rev-parse", "HEAD")
    n_gpu_layers = command_option(args.command, "--n-gpu-layers")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": list(args.command),
        "working_directory": str(working_directory),
        "git": {
            "current_commit": current_commit,
            "tracked_worktree_dirty": tracked_worktree_dirty(repo_root),
        },
        "core_guard": core_guard,
        "hardware": hardware,
        "execution_controls": {
            **local_execution,
            "model_file": model_metadata,
            "data_file": data_metadata,
            "n_gpu_layers": int(n_gpu_layers) if n_gpu_layers is not None else None,
            "accelerator_scope": (
                "local_integrated_accelerator_requested"
                if n_gpu_layers is not None and int(n_gpu_layers) > 0
                else "cpu_only_or_unspecified"
            ),
        },
        "sampling": {
            "interval_seconds": args.sample_interval,
            "memory_definition": "Sampled sum of RSS bytes for the benchmark process and its descendants.",
            "cpu_definition": "Child user plus system CPU time, normalized by wall time and logical CPU count.",
        },
        "artifacts": {
            "resource_samples": str(samples_path),
            "resource_summary": str(summary_path),
            "benchmark_summary": (
                str(benchmark_summary_path) if benchmark_summary_path else None
            ),
        },
    }
    atomic_write_json(manifest_path, manifest)

    child_cpu_before = cpu_seconds_for_children()
    wall_started = time.perf_counter()
    process = subprocess.Popen(
        args.command,
        cwd=working_directory,
        env=child_environment,
        start_new_session=True,
    )
    peak_rss_bytes = 0
    sample_count = 0
    process_tree_enumeration_complete = True
    interrupted = False
    exit_code = 1
    try:
        with samples_path.open("w", encoding="utf-8") as samples_handle:
            while True:
                elapsed = time.perf_counter() - wall_started
                sample = process_tree_sample(process.pid, elapsed)
                peak_rss_bytes = max(peak_rss_bytes, int(sample["rss_bytes"]))
                process_tree_enumeration_complete = (
                    process_tree_enumeration_complete
                    and bool(sample["tree_enumeration_complete"])
                )
                samples_handle.write(json.dumps(sample, sort_keys=True) + "\n")
                samples_handle.flush()
                sample_count += 1
                polled = process.poll()
                if polled is not None:
                    exit_code = polled
                    break
                time.sleep(args.sample_interval)
    except KeyboardInterrupt:
        interrupted = True
        stop_process_group(process)
        exit_code = 130
    finally:
        if process.poll() is None:
            stop_process_group(process)
        wall_seconds = time.perf_counter() - wall_started
        child_cpu_seconds = max(0.0, cpu_seconds_for_children() - child_cpu_before)

    processed_questions, question_count_source = resolve_processed_questions(
        benchmark_summary_path,
        args.question_count,
    )
    headline_metrics = calculate_headline_metrics(
        peak_rss_bytes=peak_rss_bytes,
        cpu_seconds=child_cpu_seconds,
        wall_seconds=wall_seconds,
        logical_cpu_count=int(hardware["logical_cpu_count"]),
        processed_questions=processed_questions,
    )
    completed = exit_code == 0 and not interrupted
    resource_summary = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "completed": completed,
        "child_exit_code": exit_code,
        "processed_questions": processed_questions,
        "processed_questions_source": question_count_source,
        "wall_seconds": wall_seconds,
        "sample_count": sample_count,
        "process_tree_enumeration_complete": process_tree_enumeration_complete,
        "headline_metrics": headline_metrics,
        "hardware": hardware,
        "execution_controls": manifest["execution_controls"],
        "definitions": {
            "peak_process_tree_rss_gib": manifest["sampling"]["memory_definition"],
            "mean_cpu_utilization_percent_of_machine": manifest["sampling"]["cpu_definition"],
            "cpu_seconds_per_processed_question": (
                "Child user plus system CPU seconds divided by questions processed during this invocation."
            ),
        },
        "measurement_notes": (
            []
            if process_tree_enumeration_complete
            else [
                "The operating system denied descendant enumeration for at least one "
                "sample; peak RAM is therefore a direct-process lower bound."
            ]
        ),
    }
    atomic_write_json(summary_path, resource_summary)

    manifest.update(
        {
            "status": "complete" if completed else ("interrupted" if interrupted else "failed"),
            "completed_at_utc": utc_now(),
            "child_exit_code": exit_code,
            "sample_count": sample_count,
        }
    )
    atomic_write_json(manifest_path, manifest)

    print(f"Wrote measurement manifest: {manifest_path}")
    print(f"Wrote resource samples: {samples_path}")
    print(f"Wrote resource summary: {summary_path}")
    print(json.dumps(headline_metrics, indent=2, sort_keys=True))
    return exit_code


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    try:
        exit_code = run_measurement(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"measurement error: {error}") from error
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
