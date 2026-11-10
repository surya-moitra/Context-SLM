"""Offline, CPU-only semantic query checking, separate from answer generation."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path


MIN_ENTAILMENT = 0.70
AMBIGUITY_MARGIN = 0.05
MAX_QUERY_CANDIDATES = 256
MAX_INTERPRETATIONS = 32
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "spm.model")


def semantic_model_identity(path):
    root = Path(path).expanduser().resolve()
    required = {"config.json", "model.safetensors", "tokenizer.json"}
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise ValueError("semantic_checker_requires_local_config_tokenizer_and_safetensors")
    files = {}
    for name in MODEL_FILES:
        source = root / name
        if not source.is_file():
            continue
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        files[name] = dict(size_bytes=source.stat().st_size, sha256=digest.hexdigest())
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(path=str(root), files=files, fingerprint=fingerprint,
                device="cpu", local_files_only=True, trust_remote_code=False,
                min_entailment=MIN_ENTAILMENT, ambiguity_margin=AMBIGUITY_MARGIN)


class LocalSemanticChecker:
    """NLI scores propose interpretations; they do not prove semantic truth."""

    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.identity = semantic_model_identity(path)
        options = dict(local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.identity["path"], **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.identity["path"], use_safetensors=True, **options).to("cpu").eval()
        labels = {str(label).lower(): int(index) for index, label in self.model.config.id2label.items()}
        if set(labels) != {"contradiction", "entailment", "neutral"}:
            raise ValueError("semantic_checker_requires_explicit_nli_label_mapping")
        self.entailment = labels["entailment"]
        self.max_length = min(512, int(self.tokenizer.model_max_length))
        self.torch = torch

    def rank(self, question, candidates):
        pairs = [(question, c["readback"]) for c in candidates]
        pairs += [(right, left) for left, right in pairs]
        scores = [None] * len(pairs)
        # Never silently truncate a requirement or a named operand.
        encoded = [self.tokenizer(a, b, truncation=False) for a, b in pairs]
        eligible = [i for i, value in enumerate(encoded) if len(value["input_ids"]) <= self.max_length]
        with self.torch.inference_mode():
            for start in range(0, len(eligible), 8):
                indices = eligible[start:start + 8]
                batch = self.tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors="pt")
                values = self.model(**batch).logits.softmax(-1)[:, self.entailment].tolist()
                for index, value in zip(indices, values):
                    if not math.isfinite(value):
                        raise ValueError("non_finite_semantic_checker_score")
                    scores[index] = float(value)
        ranked = []
        for index, plan in enumerate(candidates):
            forward, reverse = scores[index], scores[index + len(candidates)]
            if forward is not None and reverse is not None:
                ranked.append(dict(plan=plan, score=min(forward, reverse),
                                   forward_entailment=forward, reverse_entailment=reverse))
        ranked.sort(key=lambda r: (-r["score"], len(r["plan"]["steps"]),
                                   json.dumps(r["plan"]["steps"], sort_keys=True)))
        return ranked


@lru_cache(maxsize=1)
def load_semantic_checker(path):
    # Only immutable model parameters are shared, never conversation facts.
    return LocalSemanticChecker(str(Path(path).expanduser().resolve()))
