import json
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file as safetensors_load_file

from .._wginfer import DataType
from .._wginfer import DeviceType
from .._wginfer import Qwen2Meta
from .._wginfer import Qwen2Model
from ..tensor import Tensor


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self._device = device

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.meta = Qwen2Meta()
        self.meta.dtype = DataType.BF16
        self.meta.nlayer = config.get("num_hidden_layers", config.get("num_layers", 0))
        self.meta.hs = config.get("hidden_size", 0)
        self.meta.nh = config.get("num_attention_heads", 0)
        self.meta.nkvh = config.get("num_key_value_heads", self.meta.nh)
        self.meta.dh = config.get("head_dim", self.meta.hs // self.meta.nh)
        if self.meta.dh == 0:
            self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config.get("intermediate_size", 0)
        self.meta.maxseq = config.get("max_position_embeddings", 32768)
        self.meta.voc = config.get("vocab_size", 0)
        self.meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self.meta.theta = config.get("rope_theta", 1000000.0)
        self.meta.end_token = config.get("eos_token_id", 151643)

        self.model = Qwen2Model(self.meta, device, 0)
        self._load_weights(model_path)

    def _load_weights(self, model_path):
        safetensors_files = sorted(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        print(f"[wginfer] Loading Qwen2 weights from: {model_path}")
        print(f"[wginfer] Found {len(safetensors_files)} safetensors")

        def to_bf16_cpu_contig(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().cpu()
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            return t.contiguous()

        def load_wginfer_tensor_from_torch(t: torch.Tensor) -> Tensor:
            t_cpu = to_bf16_cpu_contig(t)
            lt = Tensor(shape=list(t_cpu.shape), dtype=DataType.BF16, device=self._device)
            lt.load(t_cpu.data_ptr())
            return lt

        def set_field(name: str, t: torch.Tensor):
            self.model.set_weight(name, load_wginfer_tensor_from_torch(t))

        loaded = 0
        skipped = 0

        for file_idx, file in enumerate(safetensors_files):
            print(f"[wginfer] [{file_idx + 1}/{len(safetensors_files)}] reading {file.name}")
            weights_dict = safetensors_load_file(str(file))
            print(f"[wginfer]   tensors in shard: {len(weights_dict)}")

            for key, t in weights_dict.items():
                if key == "model.embed_tokens.weight":
                    set_field("in_embed", t)
                    loaded += 1
                    continue
                if key == "lm_head.weight":
                    set_field("out_embed", t)
                    loaded += 1
                    continue
                if key == "model.norm.weight":
                    set_field("out_norm_w", t)
                    loaded += 1
                    continue

                if not key.startswith("model.layers."):
                    skipped += 1
                    continue

                parts = key.split(".")
                if len(parts) < 4:
                    skipped += 1
                    continue

                try:
                    layer_idx = int(parts[2])
                except ValueError:
                    skipped += 1
                    continue

                if layer_idx < 0 or layer_idx >= int(self.meta.nlayer):
                    skipped += 1
                    continue

                suffix = ".".join(parts[3:])
                lt = load_wginfer_tensor_from_torch(t)

                if suffix == "input_layernorm.weight":
                    self.model.set_layer_weight("attn_norm_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.q_proj.weight":
                    self.model.set_layer_weight("attn_q_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.q_proj.bias":
                    self.model.set_layer_weight("attn_q_b", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.k_proj.weight":
                    self.model.set_layer_weight("attn_k_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.k_proj.bias":
                    self.model.set_layer_weight("attn_k_b", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.v_proj.weight":
                    self.model.set_layer_weight("attn_v_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.v_proj.bias":
                    self.model.set_layer_weight("attn_v_b", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "self_attn.o_proj.weight":
                    self.model.set_layer_weight("attn_o_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "post_attention_layernorm.weight":
                    self.model.set_layer_weight("mlp_norm_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "mlp.gate_proj.weight":
                    self.model.set_layer_weight("mlp_gate_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "mlp.up_proj.weight":
                    self.model.set_layer_weight("mlp_up_w", layer_idx, lt)
                    loaded += 1
                    continue
                if suffix == "mlp.down_proj.weight":
                    self.model.set_layer_weight("mlp_down_w", layer_idx, lt)
                    loaded += 1
                    continue

                skipped += 1

            del weights_dict

        print(f"[wginfer] Done. loaded={loaded}, skipped={skipped}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        self.model.reset_cache()

        output_tokens = list(inputs)
        if len(inputs) == 0:
            return output_tokens

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = max(int(max_new_tokens), 1)

        next_token = self._infer_next(inputs, top_k, top_p, temperature)
        output_tokens.append(next_token)

        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            next_token = self._infer_next([next_token], top_k, top_p, temperature)
            output_tokens.append(next_token)

        return output_tokens

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        self.model.reset_cache()
        if len(inputs) == 0:
            return

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = max(int(max_new_tokens), 1)

        next_token = self._infer_next(inputs, top_k, top_p, temperature)
        yield next_token
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            next_token = self._infer_next([next_token], top_k, top_p, temperature)
            yield next_token

    def _infer_next(
        self,
        tokens: Sequence[int],
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> int:
        return int(
            self.model.infer(
                list(tokens),
                int(top_k),
                float(top_p),
                float(temperature),
            )
        )
