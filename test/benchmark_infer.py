import argparse
import io
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import wginfer
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from wginfer.core import RuntimeAPI

from test_utils import torch_device, wginfer_device

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RESULT_JSON_PREFIX = "__RESULT_JSON__="

PROMPTS = {
    "short": "Who are you?",
    "medium": (
        "Explain the role of KV cache in transformer decoding, and give a short "
        "step-by-step example with one prompt token and two generated tokens."
    ),
    "long": (
        "I am building a tiny LLM inference system from scratch. Please provide a "
        "concise engineering checklist that covers model loading, tensor layout, "
        "runtime abstraction, memory reuse, operator profiling, and end-to-end "
        "benchmarking. Keep the answer practical and implementation-oriented."
    ),
}

logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.ERROR)


def is_gpu_device(device_name):
    return device_name in {"nvidia", "metax"}


def parse_csv(text, caster=str):
    return [caster(x.strip()) for x in text.split(",") if x.strip()]


def resolve_model_path(model_path):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if model_path:
        model_path = os.path.expanduser(model_path)
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        return model_path
    print(f"Loading model from Hugging Face: {model_id}")
    return snapshot_download(model_id)


def load_hf_model(model_path, device_name):
    model_path = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"device_map": torch_device(device_name), "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **kwargs)
    return tokenizer, model


def load_wginfer_model(model_path, device_name, model_type):
    return wginfer.models.create_model(model_type, model_path, wginfer_device(device_name))


def load_tokenizer_only(model_path):
    model_path = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, model_path


def sync_torch(device_name):
    if is_gpu_device(device_name):
        torch.cuda.synchronize()


def sync_wginfer(device_name):
    RuntimeAPI(wginfer_device(device_name)).device_synchronize()


def build_input_ids(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(text)


def build_cases(tokenizer, prompt_names, max_new_tokens_list):
    return [
        {
            "prompt_name": prompt_name,
            "max_new_tokens": max_new_tokens,
            "input_ids": build_input_ids(tokenizer, PROMPTS[prompt_name]),
        }
        for prompt_name in prompt_names
        for max_new_tokens in max_new_tokens_list
    ]


def build_case_order(prompt_names, max_new_tokens_list):
    return [
        {
            "prompt_name": prompt_name,
            "max_new_tokens": max_new_tokens,
        }
        for prompt_name in prompt_names
        for max_new_tokens in max_new_tokens_list
    ]


def run_torch_case(tokenizer, model, input_ids, max_new_tokens, top_k, top_p, temperature, device_name):
    inputs = torch.tensor(input_ids, dtype=torch.int64, device=model.device).unsqueeze(0)
    attention_mask = torch.ones_like(inputs)

    sync_torch(device_name)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync_torch(device_name)
    out_tokens = outputs[0].tolist()
    return time.perf_counter() - start, len(out_tokens) - len(input_ids)


def run_wginfer_case(model, input_ids, max_new_tokens, top_k, top_p, temperature, device_name):
    sync_wginfer(device_name)
    start = time.perf_counter()
    out_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    sync_wginfer(device_name)
    return time.perf_counter() - start, len(out_tokens) - len(input_ids)


def benchmark_backend(backend, tokenizer, model, cases, warmup, repeat, top_k, top_p, temperature, device_name):
    rows = {}
    for case in cases:
        for _ in range(warmup):
            if backend == "torch":
                run_torch_case(
                    tokenizer,
                    model,
                    case["input_ids"],
                    case["max_new_tokens"],
                    top_k,
                    top_p,
                    temperature,
                    device_name,
                )
            else:
                run_wginfer_case(
                    model,
                    case["input_ids"],
                    case["max_new_tokens"],
                    top_k,
                    top_p,
                    temperature,
                    device_name,
                )

        latencies = []
        generated = []
        for _ in range(repeat):
            if backend == "torch":
                elapsed, new_tokens = run_torch_case(
                    tokenizer,
                    model,
                    case["input_ids"],
                    case["max_new_tokens"],
                    top_k,
                    top_p,
                    temperature,
                    device_name,
                )
            else:
                elapsed, new_tokens = run_wginfer_case(
                    model,
                    case["input_ids"],
                    case["max_new_tokens"],
                    top_k,
                    top_p,
                    temperature,
                    device_name,
                )
            latencies.append(elapsed)
            generated.append(new_tokens)

        mean_s = statistics.mean(latencies)
        rows[(case["prompt_name"], case["max_new_tokens"])] = {
            "mean_ms": mean_s * 1000.0,
            "mean_new_tokens": statistics.mean(generated),
            "tokens_per_sec": statistics.mean(generated) / mean_s if mean_s > 0 else 0.0,
        }
    return rows


def print_report(cases, torch_rows, wginfer_rows):
    print("\n=== Torch vs wGInfer Inference Benchmark ===")
    print("| Case | Torch mean(ms) | Torch tok/s | wGInfer mean(ms) | wGInfer tok/s | speedup |")
    print("|---|---:|---:|---:|---:|---:|")

    torch_total_tokens = 0.0
    wginfer_total_tokens = 0.0
    torch_total_seconds = 0.0
    wginfer_total_seconds = 0.0

    for case in cases:
        key = (case["prompt_name"], case["max_new_tokens"])
        torch_row = torch_rows[key]
        wginfer_row = wginfer_rows[key]
        speedup = torch_row["mean_ms"] / wginfer_row["mean_ms"] if wginfer_row["mean_ms"] > 0 else 0.0

        print(
            f"| {case['prompt_name']}/{case['max_new_tokens']} | {torch_row['mean_ms']:.2f} | {torch_row['tokens_per_sec']:.2f} | "
            f"{wginfer_row['mean_ms']:.2f} | {wginfer_row['tokens_per_sec']:.2f} | {speedup:.2f}x |"
        )

        torch_total_tokens += torch_row["mean_new_tokens"]
        wginfer_total_tokens += wginfer_row["mean_new_tokens"]
        torch_total_seconds += torch_row["mean_ms"] / 1000.0
        wginfer_total_seconds += wginfer_row["mean_ms"] / 1000.0

    torch_total_tok_s = torch_total_tokens / torch_total_seconds if torch_total_seconds > 0 else 0.0
    wginfer_total_tok_s = wginfer_total_tokens / wginfer_total_seconds if wginfer_total_seconds > 0 else 0.0
    overall_speedup = wginfer_total_tok_s / torch_total_tok_s if torch_total_tok_s > 0 else 0.0

    print("\n=== Throughput Summary ===")
    print(f"Torch total throughput   : {torch_total_tok_s:.2f} tok/s")
    print(f"wGInfer total throughput : {wginfer_total_tok_s:.2f} tok/s")
    print(f"Overall speedup          : {overall_speedup:.2f}x")


def build_result_payload(rows):
    return {"rows": rows}


def normalize_rows_for_json(rows):
    return {
        f"{prompt_name}::{max_new_tokens}": value
        for (prompt_name, max_new_tokens), value in rows.items()
    }


def parse_rows_from_json(rows):
    parsed = {}
    for key, value in rows.items():
        prompt_name, max_new_tokens = key.split("::", 1)
        parsed[(prompt_name, int(max_new_tokens))] = value
    return parsed


def run_backend_only(args):
    prompt_names = parse_csv(args.prompts)
    max_new_tokens_list = parse_csv(args.max_new_tokens, int)

    for name in prompt_names:
        if name not in PROMPTS:
            raise ValueError(f"Unknown prompt preset: {name}. Valid keys: {list(PROMPTS.keys())}")

    if args.backend_only == "torch":
        tokenizer, model = load_hf_model(args.model, args.device)
        cases = build_cases(tokenizer, prompt_names, max_new_tokens_list)
        rows = benchmark_backend(
            "torch",
            tokenizer,
            model,
            cases,
            args.warmup,
            args.repeat,
            args.top_k,
            args.top_p,
            args.temperature,
            args.device,
        )
        print(RESULT_JSON_PREFIX + json.dumps(build_result_payload(normalize_rows_for_json(rows)), ensure_ascii=False))
        return

    if args.backend_only == "wginfer":
        tokenizer, model_path = load_tokenizer_only(args.model)
        model = load_wginfer_model(model_path, args.device, args.model_type)
        cases = build_cases(tokenizer, prompt_names, max_new_tokens_list)
        rows = benchmark_backend(
            "wginfer",
            tokenizer,
            model,
            cases,
            args.warmup,
            args.repeat,
            args.top_k,
            args.top_p,
            args.temperature,
            args.device,
        )
        print(RESULT_JSON_PREFIX + json.dumps(build_result_payload(normalize_rows_for_json(rows)), ensure_ascii=False))
        return

    raise ValueError(f"Unknown backend_only value: {args.backend_only}")


def run_backend_subprocess(args, backend, model_path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend-only",
        backend,
        "--model-type",
        args.model_type,
        "--device",
        args.device,
        "--prompts",
        args.prompts,
        "--max-new-tokens",
        args.max_new_tokens,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--temperature",
        str(args.temperature),
        "--model",
        str(model_path),
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} subprocess failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    lines = completed.stdout.splitlines()
    for line in reversed(lines):
        if line.startswith(RESULT_JSON_PREFIX):
            payload = json.loads(line[len(RESULT_JSON_PREFIX):])
            return parse_rows_from_json(payload["rows"])

    raise RuntimeError(
        f"Failed to parse {backend} subprocess result.\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Torch vs wGInfer inference throughput.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--model-type", default="qwen2", choices=wginfer.models.supported_model_types(), type=str)
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax"], type=str)
    parser.add_argument("--prompts", default="short,medium,long", type=str)
    parser.add_argument("--max-new-tokens", default="32,64,128", type=str)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--top-k", default=1, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--backend-only", choices=["torch", "wginfer"], default=None)
    args = parser.parse_args()

    if args.backend_only is not None:
        run_backend_only(args)
        return

    model_path = resolve_model_path(args.model)
    prompt_names = parse_csv(args.prompts)
    max_new_tokens_list = parse_csv(args.max_new_tokens, int)
    for name in prompt_names:
        if name not in PROMPTS:
            raise ValueError(f"Unknown prompt preset: {name}. Valid keys: {list(PROMPTS.keys())}")
    cases = build_case_order(prompt_names, max_new_tokens_list)

    torch_rows = run_backend_subprocess(args, "torch", model_path)
    wginfer_rows = run_backend_subprocess(args, "wginfer", model_path)

    print_report(cases, torch_rows, wginfer_rows)


if __name__ == "__main__":
    main()
