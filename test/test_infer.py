import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
import os
import time
import wginfer
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
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


def topk_pairs_from_logits(logits, k=10):
    values, indices = torch.topk(logits.to(torch.float32), k=min(int(k), logits.numel()))
    return [(int(idx), float(val)) for idx, val in zip(indices.tolist(), values.tolist())]


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


def hf_first_step_logits(prompt, tokenizer, hf_model):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(input_content)

    hf_inputs = torch.tensor(input_ids, dtype=torch.int64, device=hf_model.device).unsqueeze(0)
    hf_attention_mask = torch.ones_like(hf_inputs)
    with torch.no_grad():
        hf_logits = hf_model(
            hf_inputs,
            attention_mask=hf_attention_mask,
        ).logits[0, -1].detach().cpu()
    return input_ids, hf_logits


def debug_first_step_logits(input_ids, hf_logits, wginfer_model):
    if hasattr(wginfer_model, "forward_logits_host"):
        wg_logits = wginfer_model.forward_logits_host(input_ids, max_layers=32)[-1].detach().cpu().to(torch.float32)
    else:
        raise RuntimeError("Current model does not expose forward_logits_host for debug")

    print("\n=== Debug First Step Logits ===\n")
    print("HF top-10:")
    print(topk_pairs_from_logits(hf_logits, 10))
    print("\nwGInfer top-10:")
    print(topk_pairs_from_logits(wg_logits, 10))
    print("\nHF argmax:", int(torch.argmax(hf_logits).item()))
    print("wGInfer argmax:", int(torch.argmax(wg_logits).item()))


def tensor_summary(name, x):
    flat = x.reshape(-1).to(torch.float32)
    preview = flat[:8].tolist()
    print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} mean={float(flat.mean()):.6f} std={float(flat.std()):.6f}")
    print(f"{name}[:8]: {preview}")


def hf_debug_hidden_states(prompt, tokenizer, hf_model):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(input_content)
    hf_inputs = torch.tensor(input_ids, dtype=torch.int64, device=hf_model.device).unsqueeze(0)
    hf_attention_mask = torch.ones_like(hf_inputs)
    with torch.no_grad():
        outputs = hf_model.model(
            hf_inputs,
            attention_mask=hf_attention_mask,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    return input_ids, {
        "embedding": hidden_states[0][0].detach().cpu(),
        "layer1": hidden_states[1][0].detach().cpu(),
        "layer4": hidden_states[4][0].detach().cpu(),
    }


def debug_layers(input_ids, hf_hidden_states, wginfer_model):
    if not hasattr(wginfer_model, "forward_prefix_host"):
        raise RuntimeError("Current model does not expose forward_prefix_host for debug")

    wg_hidden_states = {
        "embedding": wginfer_model.forward_prefix_host(input_ids, max_layers=0).detach().cpu().to(torch.float32),
        "layer1": wginfer_model.forward_prefix_host(input_ids, max_layers=1).detach().cpu().to(torch.float32),
        "layer4": wginfer_model.forward_prefix_host(input_ids, max_layers=4).detach().cpu().to(torch.float32),
    }

    print("\n=== Debug Layer States ===\n")
    for name in ("embedding", "layer1", "layer4"):
        hf_x = hf_hidden_states[name].to(torch.float32)
        wg_x = wg_hidden_states[name].to(torch.float32)
        tensor_summary(f"HF {name}", hf_x)
        tensor_summary(f"wGInfer {name}", wg_x)
        diff = (hf_x - wg_x).abs()
        print(
            f"{name} diff: mean_abs={float(diff.mean()):.6f} max_abs={float(diff.max()):.6f}\n"
        )


def debug_linear_layer(model, input_ids):
    if not hasattr(model, "debug_linear_layer_host"):
        raise RuntimeError("Current model does not expose debug_linear_layer_host")

    tensors = model.debug_linear_layer_host(input_ids)
    print("\n=== Debug Linear Layer Internals ===\n")
    for name in (
        "x_norm",
        "qkv_proj",
        "z_proj",
        "a_proj",
        "b_proj",
        "mixed_qkv",
        "q_ready",
        "k_ready",
        "v_ready",
        "z_ready",
        "g_ready",
        "beta_ready",
        "core_attn",
        "gated",
        "attn_proj",
        "x_attn",
    ):
        tensor_summary(name, tensors[name])
        print("")


def hf_debug_linear_layer(prompt, tokenizer, hf_model):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(input_content)
    input_tensor = torch.tensor(input_ids, dtype=torch.int64, device=hf_model.device).unsqueeze(0)

    text_model = hf_model.model
    layer = text_model.layers[0]
    linear_attn = layer.linear_attn

    with torch.no_grad():
        hidden_states = text_model.embed_tokens(input_tensor)
        x_norm = layer.input_layernorm(hidden_states)
        mixed_qkv = linear_attn.in_proj_qkv(x_norm)
        z = linear_attn.in_proj_z(x_norm).reshape(hidden_states.shape[0], hidden_states.shape[1], -1, linear_attn.head_v_dim)
        b = linear_attn.in_proj_b(x_norm)
        a = linear_attn.in_proj_a(x_norm)

        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        mixed_qkv_after_conv = F.silu(linear_attn.conv1d(mixed_qkv_t)[:, :, : hidden_states.shape[1]])
        mixed_qkv = mixed_qkv_after_conv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim],
            dim=-1,
        )
        query = query.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, linear_attn.head_k_dim)
        key = key.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, linear_attn.head_k_dim)
        value = value.reshape(hidden_states.shape[0], hidden_states.shape[1], -1, linear_attn.head_v_dim)

        beta = b.sigmoid()
        g = -linear_attn.A_log.float().exp() * F.softplus(a.float() + linear_attn.dt_bias)
        if linear_attn.num_v_heads // linear_attn.num_k_heads > 1:
            query = query.repeat_interleave(linear_attn.num_v_heads // linear_attn.num_k_heads, dim=2)
            key = key.repeat_interleave(linear_attn.num_v_heads // linear_attn.num_k_heads, dim=2)

    return input_ids, {
        "x_norm": x_norm[0].detach().cpu().to(torch.float32),
        "qkv_proj": linear_attn.in_proj_qkv(x_norm)[0].detach().cpu().to(torch.float32),
        "z_proj": linear_attn.in_proj_z(x_norm)[0].detach().cpu().to(torch.float32),
        "a_proj": a[0].detach().cpu().to(torch.float32),
        "b_proj": b[0].detach().cpu().to(torch.float32),
        "mixed_qkv": mixed_qkv[0].detach().cpu().to(torch.float32),
        "q_ready": query[0].detach().cpu().to(torch.float32),
        "k_ready": key[0].detach().cpu().to(torch.float32),
        "v_ready": value[0].detach().cpu().to(torch.float32),
        "z_ready": z[0].reshape(-1, linear_attn.head_v_dim).detach().cpu().to(torch.float32),
        "g_ready": g[0].detach().cpu().to(torch.float32),
        "beta_ready": beta[0].detach().cpu().to(torch.float32),
    }


def debug_linear_layer_compare(hf_tensors, wg_model, input_ids):
    if not hasattr(wg_model, "debug_linear_layer_host"):
        raise RuntimeError("Current model does not expose debug_linear_layer_host")

    wg_tensors = wg_model.debug_linear_layer_host(input_ids)
    print("\n=== Debug Linear Layer Internals ===\n")
    for name in (
        "x_norm",
        "qkv_proj",
        "z_proj",
        "a_proj",
        "b_proj",
        "mixed_qkv",
        "q_ready",
        "k_ready",
        "v_ready",
        "z_ready",
        "g_ready",
        "beta_ready",
    ):
        hf_x = hf_tensors[name].to(torch.float32)
        wg_x = wg_tensors[name].to(torch.float32)
        tensor_summary(f"HF {name}", hf_x)
        tensor_summary(f"wGInfer {name}", wg_x)
        diff = (hf_x - wg_x).abs()
        print(f"{name} diff: mean_abs={float(diff.mean()):.6f} max_abs={float(diff.max()):.6f}\n")


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
    parser.add_argument("--debug-first-step", action="store_true")
    parser.add_argument("--debug-layers", action="store_true")
    parser.add_argument("--debug-linear-layer", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    do_sample = args.do_sample
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0
        do_sample = False

    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    debug_input_ids = None
    debug_hf_logits = None
    debug_hf_hidden_states = None
    debug_hf_linear_tensors = None
    if args.debug_first_step:
        debug_input_ids, debug_hf_logits = hf_first_step_logits(args.prompt, tokenizer, model)
    if args.debug_layers:
        debug_input_ids, debug_hf_hidden_states = hf_debug_hidden_states(args.prompt, tokenizer, model)
    if args.debug_linear_layer:
        debug_input_ids, debug_hf_linear_tensors = hf_debug_linear_layer(args.prompt, tokenizer, model)

    # Example prompt
    start_time = time.time()
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        do_sample=do_sample,
    )
    end_time = time.time()

    del model
    gc.collect()
    if args.device == "nvidia":
        # Release PyTorch caching allocator blocks before running WGINFER in the same process.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    model = load_wginfer_model(model_path, args.device, args.model_type)
    if args.debug_first_step:
        debug_first_step_logits(debug_input_ids, debug_hf_logits, model)
    if args.debug_layers:
        debug_layers(debug_input_ids, debug_hf_hidden_states, model)
    if args.debug_linear_layer:
        debug_linear_layer_compare(debug_hf_linear_tensors, model, debug_input_ids)
    start_time = time.time()
    wginfer_tokens, wginfer_output = wginfer_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        do_sample=do_sample,
    )

    end_time = time.time()

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(wginfer_tokens)
    print("\nContents:")
    print(wginfer_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        assert wginfer_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")
