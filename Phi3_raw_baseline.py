#!/usr/bin/env python3

"""Raw local Phi-3 chat baseline.

This file intentionally does not import or use the PRAGMOS context layer. It is
the clean small-LLM baseline for comparing Phi-3 by itself against Phi-3 with
PRAGMOS memory/context injection.
"""

import argparse
import os
import time


DEFAULT_LLM_PATH = os.environ.get(
    "PHI3_GGUF_PATH",
    "./../phi3-gguf/Phi-3-mini-4k-instruct-q4.gguf",
)
DEFAULT_CONTEXT_LENGTH = int(os.environ.get("PHI3_CONTEXT_LENGTH", "2048"))
DEFAULT_NUM_THREADS = int(os.environ.get("PHI3_NUM_THREADS", "8"))
DEFAULT_GPU_LAYERS = int(os.environ.get("PHI3_GPU_LAYERS", "40"))
DEFAULT_OFFLOAD_KQV = os.environ.get("PHI3_OFFLOAD_KQV", "1").lower() not in {
    "0",
    "false",
    "no",
}
DEFAULT_MAX_TOKENS = int(os.environ.get("PHI3_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("PHI3_TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.environ.get("PHI3_TOP_P", "1.0"))
DEFAULT_REPEAT_PENALTY = float(os.environ.get("PHI3_REPEAT_PENALTY", "1.1"))
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly and "
    "concisely. If the answer cannot be determined, say you do not know."
)


class Phi3RawChat:
    """Thin llama.cpp wrapper for raw Phi-3 chat completion."""

    def __init__(
        self,
        model_path=DEFAULT_LLM_PATH,
        n_ctx=DEFAULT_CONTEXT_LENGTH,
        n_threads=DEFAULT_NUM_THREADS,
        n_gpu_layers=DEFAULT_GPU_LAYERS,
        offload_kqv=DEFAULT_OFFLOAD_KQV,
        seed=42,
        verbose=False,
    ):
        from llama_cpp import Llama

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.llm = self._load_llama(
            Llama=Llama,
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            offload_kqv=offload_kqv,
            seed=seed,
            verbose=verbose,
        )

    def _load_llama(
        self,
        Llama,
        model_path,
        n_ctx,
        n_threads,
        n_gpu_layers,
        offload_kqv,
        seed,
        verbose,
    ):
        kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "offload_kqv": offload_kqv,
            "verbose": verbose,
        }
        if seed is not None:
            kwargs["seed"] = seed

        try:
            return Llama(**kwargs)
        except TypeError:
            kwargs.pop("verbose", None)
            try:
                return Llama(**kwargs)
            except TypeError:
                kwargs.pop("offload_kqv", None)
                try:
                    return Llama(**kwargs)
                except TypeError:
                    kwargs.pop("seed", None)
                    return Llama(**kwargs)

    def format_prompt(self, user_input, system_prompt=DEFAULT_SYSTEM_PROMPT):
        if system_prompt:
            return (
                f"<|system|>\n{system_prompt}<|end|>\n"
                f"<|user|>\n{user_input}<|end|>\n"
                f"<|assistant|>\n"
            )
        return f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

    def count_tokens(self, text):
        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            return max(1, (len(text or "") + 3) // 4)

    def complete(
        self,
        prompt,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        repeat_penalty=DEFAULT_REPEAT_PENALTY,
        stop=None,
    ):
        stop = stop or ["<|end|>", "<|user|>", "<|system|>"]
        started_at = time.perf_counter()
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=False,
        )
        elapsed_seconds = time.perf_counter() - started_at
        text = output["choices"][0]["text"].strip()
        return {
            "text": text,
            "elapsed_seconds": elapsed_seconds,
            "prompt_tokens_estimate": self.count_tokens(prompt),
            "completion_tokens_estimate": self.count_tokens(text),
            "raw_output": output,
        }

    def chat(
        self,
        user_input,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        repeat_penalty=DEFAULT_REPEAT_PENALTY,
    ):
        prompt = self.format_prompt(
            user_input=user_input,
            system_prompt=system_prompt,
        )
        return self.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run raw local Phi-3 chat.")
    parser.add_argument("--model-path", default=DEFAULT_LLM_PATH)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--n-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--n-gpu-layers", type=int, default=DEFAULT_GPU_LAYERS)
    parser.add_argument(
        "--no-offload-kqv",
        action="store_true",
        help="Disable llama.cpp K/Q/V offload. Useful for CPU-only or restricted environments.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repeat-penalty", type=float, default=DEFAULT_REPEAT_PENALTY)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt", help="Single prompt to answer. Omit for REPL mode.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    chat_model = Phi3RawChat(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        offload_kqv=not args.no_offload_kqv,
        seed=args.seed,
        verbose=args.verbose,
    )

    if args.prompt:
        result = chat_model.chat(
            user_input=args.prompt,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )
        print(result["text"])
        return

    print("Raw Phi-3 ready. Type 'exit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        result = chat_model.chat(
            user_input=user_input,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
        )
        print("Bot:", result["text"])


if __name__ == "__main__":
    main()
