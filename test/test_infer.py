from test_utils import *

import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import subprocess
from pathlib import Path
import time
import wginfer
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RESULT_JSON_PREFIX = "__RESULT_JSON__="


def resolve_model_path(model_path=None):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if model_path:
        model_path = os.path.expanduser(model_path)
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        return model_path
    print(f"Loading model from Hugging Face: {model_id}")
    return snapshot_download(model_id)


def load_hf_model(model_path=None, device_name="cpu"):
    model_path = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt,
    tokenizer,
    model,
    max_new_tokens=128,
    top_p=1.0,
    top_k=1,
    temperature=1.0,
    do_sample=False,
):
    # 1、user prompt ----》model format prompt
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,    # 在最后自动补一个assistant即将开始的问答提示头
        tokenize=False,  # False：只返回格式化后的字符串，不返回token ids；True：返回token ids/tokenizer输出
    )
    # 2、mdoel format prompt ----》token ids[batch_size，seqlen]
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(inputs)
    # 3、close grad，infer only
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 4、decode：output_ids[batch_size, total_len]->text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_wginfer_model(model_path, device_name, model_type):
    model = wginfer.models.create_model(model_type, model_path, wginfer_device(device_name))
    return model


def load_tokenizer_only(model_path=None):
    resolved = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    return tokenizer, resolved


def wginfer_infer(
    prompt,
    tokenizer,
    model,
    max_new_tokens=128,
    top_p=1.0,
    top_k=1,
    temperature=1.0,
    do_sample=False,
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    if not do_sample:
        top_k, top_p, temperature = 1, 1.0, 1.0
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


def build_result_payload(backend, model_path, tokens, output, elapsed):
    return {
        "backend": backend,
        "model_path": str(model_path),
        "tokens": tokens,
        "output": output,
        "elapsed": elapsed,
    }


def run_backend_only(args):
    if args.backend_only == "hf":
        tokenizer, model, model_path = load_hf_model(args.model, args.device)
        start_time = time.time()
        tokens, output = hf_infer(
            args.prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
        elapsed = time.time() - start_time
        print(RESULT_JSON_PREFIX + json.dumps(build_result_payload("hf", model_path, tokens, output, elapsed), ensure_ascii=False))
        return

    if args.backend_only == "wginfer":
        tokenizer, model_path = load_tokenizer_only(args.model)
        model = load_wginfer_model(model_path, args.device, args.model_type)
        start_time = time.time()
        tokens, output = wginfer_infer(
            args.prompt,
            tokenizer,
            model,
            max_new_tokens=args.max_steps,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
        elapsed = time.time() - start_time
        print(RESULT_JSON_PREFIX + json.dumps(build_result_payload("wginfer", model_path, tokens, output, elapsed), ensure_ascii=False))
        return

    raise ValueError(f"Unknown backend_only value: {args.backend_only}")


def run_backend_subprocess(args, backend, model_path_override=None):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend-only",
        backend,
        "--device",
        args.device,
        "--model-type",
        args.model_type,
        "--prompt",
        args.prompt,
        "--max_steps",
        str(args.max_steps),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--temperature",
        str(args.temperature),
    ]
    if args.do_sample:
        cmd.append("--do-sample")
    if model_path_override is not None:
        cmd.extend(["--model", str(model_path_override)])
    elif args.model is not None:
        cmd.extend(["--model", args.model])

    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} subprocess failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    lines = completed.stdout.splitlines()
    for line in reversed(lines):
        if line.startswith(RESULT_JSON_PREFIX):
            return json.loads(line[len(RESULT_JSON_PREFIX):])

    raise RuntimeError(
        f"Failed to parse {backend} subprocess result.\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )


def print_backend_result(title, tokens, output, elapsed):
    print(f"\n=== {title} ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {elapsed:.2f}s\n")


def run_split_process_mode(args):
    hf_result = run_backend_subprocess(args, "hf")
    print_backend_result("Answer", hf_result["tokens"], hf_result["output"], hf_result["elapsed"])

    wginfer_result = run_backend_subprocess(args, "wginfer", model_path_override=hf_result["model_path"])
    print_backend_result("Your Result", wginfer_result["tokens"], wginfer_result["output"], wginfer_result["elapsed"])

    if args.test:
        assert wginfer_result["tokens"] == hf_result["tokens"]
        print("\033[92mTest passed!\033[0m\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--model-type", default="qwen2", choices=wginfer.models.supported_model_types(), type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--backend-only", choices=["hf", "wginfer"], default=None)

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    do_sample = args.do_sample
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0
        do_sample = False
    args.top_p = top_p
    args.top_k = top_k
    args.temperature = temperature
    args.do_sample = do_sample

    if args.backend_only is not None:
        run_backend_only(args)
    else:
        run_split_process_mode(args)
