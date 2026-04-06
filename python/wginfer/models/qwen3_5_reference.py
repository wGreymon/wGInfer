from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load_file

from .._wginfer.core import DataType
from .._wginfer.core import DeviceType
from .._wginfer.core import MemcpyKind
from .._wginfer.core import Ops
from .._wginfer.core import RuntimeAPI
from .._wginfer.core import Tensor
from ._config import ModelConfigView
from ._config import UnsupportedModelConfigError
from ._config import load_model_config
from ._config import resolve_model_config
from ._config import rope_theta_from_text_config

try:
    from .._wginfer.models import Qwen3_5LayerType as _RuntimeQwen3_5LayerType
    from .._wginfer.models import Qwen3_5Meta as _RuntimeQwen3_5Meta
    from .._wginfer.models import Qwen3_5Model as _RuntimeQwen3_5Model
except ImportError:
    _RuntimeQwen3_5LayerType = None
    _RuntimeQwen3_5Meta = None
    _RuntimeQwen3_5Model = None


def is_qwen3_5_config(view: ModelConfigView) -> bool:
    candidates = {view.model_type, view.text_model_type}
    if any(candidate in {"qwen3_5", "qwen3_5_text"} for candidate in candidates if candidate):
        return True
    return any(arch.startswith("Qwen3_5") for arch in view.architectures)


@dataclass(frozen=True)
class Qwen3_5LayerSpec:
    index: int
    layer_type: str


@dataclass(frozen=True)
class Qwen3_5TextMeta:
    dtype: DataType
    nlayer: int
    hs: int
    nh: int
    nkvh: int
    dh: int
    di: int
    maxseq: int
    voc: int
    epsilon: float
    theta: float
    end_token: int
    tie_word_embeddings: bool
    full_attention_interval: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    partial_rotary_factor: float
    layer_types: tuple[str, ...]


@dataclass(frozen=True)
class Qwen3_5WeightIndex:
    total_size: int
    shard_files: tuple[str, ...]
    weight_map: dict[str, str]
    language_model_weight_count: int
    language_model_global_weight_count: int
    vision_weight_count: int
    mtp_weight_count: int


def parse_qwen3_5_text_meta(text_config: dict[str, Any]) -> Qwen3_5TextMeta:
    hs = int(text_config.get("hidden_size", 0))
    nh = int(text_config.get("num_attention_heads", 0))
    dh = int(text_config.get("head_dim", hs // nh if nh else 0))
    layer_types_cfg = text_config.get("layer_types")
    layer_types = tuple(str(x) for x in layer_types_cfg) if isinstance(layer_types_cfg, list) else ()

    return Qwen3_5TextMeta(
        dtype=DataType.BF16,
        nlayer=int(text_config.get("num_hidden_layers", 0)),
        hs=hs,
        nh=nh,
        nkvh=int(text_config.get("num_key_value_heads", nh)),
        dh=dh,
        di=int(text_config.get("intermediate_size", 0)),
        maxseq=int(text_config.get("max_position_embeddings", 0)),
        voc=int(text_config.get("vocab_size", 0)),
        epsilon=float(text_config.get("rms_norm_eps", 1e-6)),
        theta=rope_theta_from_text_config(text_config, 10_000_000.0),
        end_token=int(text_config.get("eos_token_id", -1)),
        tie_word_embeddings=bool(text_config.get("tie_word_embeddings", False)),
        full_attention_interval=int(text_config.get("full_attention_interval", 0)),
        linear_num_key_heads=int(text_config.get("linear_num_key_heads", 0)),
        linear_num_value_heads=int(text_config.get("linear_num_value_heads", 0)),
        linear_key_head_dim=int(text_config.get("linear_key_head_dim", 0)),
        linear_value_head_dim=int(text_config.get("linear_value_head_dim", 0)),
        linear_conv_kernel_dim=int(text_config.get("linear_conv_kernel_dim", 0)),
        partial_rotary_factor=float(text_config.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)),
        layer_types=layer_types,
    )


def load_qwen3_5_weight_index(model_path: str | Path) -> Qwen3_5WeightIndex:
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"model.safetensors.index.json not found in {model_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid weight_map in {index_path}")

    shard_files = tuple(sorted({str(v) for v in weight_map.values()}))
    language_model_weight_count = 0
    language_model_global_weight_count = 0
    vision_weight_count = 0
    mtp_weight_count = 0
    for key in weight_map:
        if key.startswith("model.language_model.layers."):
            language_model_weight_count += 1
        elif key.startswith("model.language_model."):
            language_model_global_weight_count += 1
        elif key.startswith("model.visual."):
            vision_weight_count += 1
        elif key.startswith("mtp."):
            mtp_weight_count += 1

    return Qwen3_5WeightIndex(
        total_size=int(metadata.get("total_size", 0)),
        shard_files=shard_files,
        weight_map={str(k): str(v) for k, v in weight_map.items()},
        language_model_weight_count=language_model_weight_count,
        language_model_global_weight_count=language_model_global_weight_count,
        vision_weight_count=vision_weight_count,
        mtp_weight_count=mtp_weight_count,
    )


class Qwen3_5:
    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
    ):
        model_path = Path(model_path)
        config = load_model_config(model_path)
        view = resolve_model_config(config)
        if not is_qwen3_5_config(view):
            raise UnsupportedModelConfigError(
                f"Model at {model_path} is not recognized as a Qwen3.5 model."
            )

        self.model_path = model_path
        self.device = device
        self.text_only = True
        self.config_view = view
        self.meta = parse_qwen3_5_text_meta(view.text_config)
        self.layers = tuple(
            Qwen3_5LayerSpec(index=i, layer_type=layer_type)
            for i, layer_type in enumerate(self.meta.layer_types)
        )
        self.weight_index = load_qwen3_5_weight_index(model_path)
        self._validate_weight_index()
        self.runtime_meta = None
        self.runtime_model = None
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._host_weight_cache: dict[str, torch.Tensor] = {}
        self._tensor_weight_cache: dict[tuple[str, bool], Tensor] = {}
        self._runtime_api = RuntimeAPI(device)
        self._debug_linear_layer_data: dict[str, torch.Tensor] | None = None
        self._argmax_idx: Tensor | None = None
        self._argmax_val: Tensor | None = None
        self._runtime_weights_loaded = False
        self.reset_cache()
        self._init_runtime_skeleton()

    def _init_runtime_skeleton(self) -> None:
        if _RuntimeQwen3_5Meta is None or _RuntimeQwen3_5Model is None or _RuntimeQwen3_5LayerType is None:
            return

        runtime_meta = _RuntimeQwen3_5Meta()
        runtime_meta.dtype = self.meta.dtype
        runtime_meta.nlayer = self.meta.nlayer
        runtime_meta.hs = self.meta.hs
        runtime_meta.nh = self.meta.nh
        runtime_meta.nkvh = self.meta.nkvh
        runtime_meta.dh = self.meta.dh
        runtime_meta.di = self.meta.di
        runtime_meta.maxseq = self.meta.maxseq
        runtime_meta.voc = self.meta.voc
        runtime_meta.epsilon = self.meta.epsilon
        runtime_meta.theta = self.meta.theta
        runtime_meta.end_token = self.meta.end_token
        runtime_meta.tie_word_embeddings = self.meta.tie_word_embeddings
        runtime_meta.full_attention_interval = self.meta.full_attention_interval
        runtime_meta.linear_num_key_heads = self.meta.linear_num_key_heads
        runtime_meta.linear_num_value_heads = self.meta.linear_num_value_heads
        runtime_meta.linear_key_head_dim = self.meta.linear_key_head_dim
        runtime_meta.linear_value_head_dim = self.meta.linear_value_head_dim
        runtime_meta.linear_conv_kernel_dim = self.meta.linear_conv_kernel_dim
        runtime_meta.partial_rotary_factor = self.meta.partial_rotary_factor

        layer_type_map = {
            "linear_attention": _RuntimeQwen3_5LayerType.LINEAR_ATTENTION,
            "full_attention": _RuntimeQwen3_5LayerType.FULL_ATTENTION,
        }
        runtime_layer_types = [layer_type_map[layer.layer_type] for layer in self.layers]
        self.runtime_meta = runtime_meta
        self.runtime_model = _RuntimeQwen3_5Model(runtime_meta, runtime_layer_types, self.device, 0)

    @staticmethod
    def _runtime_norm_weight_needs_offset(key: str) -> bool:
        if key == "model.language_model.norm.weight":
            return True
        return key.endswith((
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ))

    def _runtime_tensor_from_torch(
        self,
        tensor: torch.Tensor,
        *,
        add_one: bool = False,
        dtype: DataType | None = None,
    ) -> Tensor:
        host_tensor = tensor.detach().cpu().contiguous()
        if add_one:
            host_tensor = host_tensor.to(torch.float32) + 1.0
        return self._to_wginfer_tensor(host_tensor, dtype=dtype)

    def _load_runtime_weights(self) -> None:
        if self.runtime_model is None or self._runtime_weights_loaded:
            return

        loaded = 0
        skipped = 0

        def set_global(name: str, tensor: torch.Tensor, *, add_one: bool = False) -> None:
            self.runtime_model.set_weight(name, self._runtime_tensor_from_torch(tensor, add_one=add_one))

        def set_layer(
            name: str,
            layer_idx: int,
            tensor: torch.Tensor,
            *,
            add_one: bool = False,
            dtype: DataType | None = None,
        ) -> None:
            self.runtime_model.set_layer_weight(
                name,
                layer_idx,
                self._runtime_tensor_from_torch(tensor, add_one=add_one, dtype=dtype),
            )

        for shard_name in self.weight_index.shard_files:
            shard = self._load_shard(shard_name)
            for key, tensor in shard.items():
                if key == "model.language_model.embed_tokens.weight":
                    set_global("in_embed", tensor)
                    loaded += 1
                    continue
                if key in {"lm_head.weight", "model.language_model.lm_head.weight"}:
                    set_global("out_embed", tensor)
                    loaded += 1
                    continue
                if key == "model.language_model.norm.weight":
                    set_global("out_norm_w", tensor, add_one=True)
                    loaded += 1
                    continue

                if not key.startswith("model.language_model.layers."):
                    skipped += 1
                    continue

                parts = key.split(".")
                if len(parts) < 5:
                    skipped += 1
                    continue

                try:
                    layer_idx = int(parts[3])
                except ValueError:
                    skipped += 1
                    continue
                if layer_idx < 0 or layer_idx >= self.meta.nlayer:
                    skipped += 1
                    continue

                suffix = ".".join(parts[4:])
                add_one = self._runtime_norm_weight_needs_offset(key)

                if suffix == "input_layernorm.weight":
                    set_layer("attn_norm_w", layer_idx, tensor, add_one=add_one)
                    loaded += 1
                    continue
                if suffix == "post_attention_layernorm.weight":
                    set_layer("mlp_norm_w", layer_idx, tensor, add_one=add_one)
                    loaded += 1
                    continue
                if suffix == "mlp.gate_proj.weight":
                    set_layer("mlp_gate_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "mlp.up_proj.weight":
                    set_layer("mlp_up_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "mlp.down_proj.weight":
                    set_layer("mlp_down_w", layer_idx, tensor)
                    loaded += 1
                    continue

                if suffix == "self_attn.q_proj.weight":
                    set_layer("full_attn_q_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "self_attn.k_proj.weight":
                    set_layer("full_attn_k_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "self_attn.v_proj.weight":
                    set_layer("full_attn_v_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "self_attn.o_proj.weight":
                    set_layer("full_attn_o_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "self_attn.q_norm.weight":
                    set_layer("full_attn_q_norm_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "self_attn.k_norm.weight":
                    set_layer("full_attn_k_norm_w", layer_idx, tensor)
                    loaded += 1
                    continue

                if suffix == "linear_attn.in_proj_qkv.weight":
                    set_layer("linear_attn_qkv_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.in_proj_z.weight":
                    set_layer("linear_attn_z_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.out_proj.weight":
                    set_layer("linear_attn_o_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.in_proj_a.weight":
                    set_layer("linear_attn_a_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.in_proj_b.weight":
                    set_layer("linear_attn_b_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.norm.weight":
                    set_layer("linear_attn_norm_w", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.dt_bias":
                    set_layer("linear_attn_dt_bias", layer_idx, tensor)
                    loaded += 1
                    continue
                if suffix == "linear_attn.A_log":
                    set_layer("linear_attn_a_log", layer_idx, tensor, dtype=DataType.F32)
                    loaded += 1
                    continue
                if suffix == "linear_attn.conv1d.weight":
                    set_layer("linear_attn_conv_w", layer_idx, tensor)
                    loaded += 1
                    continue

                skipped += 1

        self._runtime_weights_loaded = True
        print(f"[wginfer] Qwen3.5 runtime weights ready. loaded={loaded}, skipped={skipped}")

    def _validate_weight_index(self) -> None:
        weight_map = self.weight_index.weight_map
        missing = []
        required_global = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
        ]
        for key in required_global:
            if key not in weight_map:
                missing.append(key)

        for layer in self.layers:
            layer_prefix = f"model.language_model.layers.{layer.index}."
            common = [
                layer_prefix + "input_layernorm.weight",
                layer_prefix + "post_attention_layernorm.weight",
                layer_prefix + "mlp.gate_proj.weight",
                layer_prefix + "mlp.up_proj.weight",
                layer_prefix + "mlp.down_proj.weight",
            ]
            if layer.layer_type == "full_attention":
                required = common + [
                    layer_prefix + "self_attn.q_proj.weight",
                    layer_prefix + "self_attn.k_proj.weight",
                    layer_prefix + "self_attn.v_proj.weight",
                    layer_prefix + "self_attn.o_proj.weight",
                    layer_prefix + "self_attn.q_norm.weight",
                    layer_prefix + "self_attn.k_norm.weight",
                ]
            elif layer.layer_type == "linear_attention":
                required = common + [
                    layer_prefix + "linear_attn.in_proj_qkv.weight",
                    layer_prefix + "linear_attn.in_proj_z.weight",
                    layer_prefix + "linear_attn.out_proj.weight",
                    layer_prefix + "linear_attn.in_proj_a.weight",
                    layer_prefix + "linear_attn.in_proj_b.weight",
                    layer_prefix + "linear_attn.norm.weight",
                    layer_prefix + "linear_attn.dt_bias",
                    layer_prefix + "linear_attn.A_log",
                    layer_prefix + "linear_attn.conv1d.weight",
                ]
            else:
                raise UnsupportedModelConfigError(
                    f"Unsupported Qwen3.5 layer type `{layer.layer_type}` in {self.model_path}."
                )

            for key in required:
                if key not in weight_map:
                    missing.append(key)

        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += f", ... (+{len(missing) - 8} more)"
            raise UnsupportedModelConfigError(
                f"Qwen3.5 checkpoint at {self.model_path} is missing expected language-model weights: {preview}"
            )

    def summary(self) -> dict[str, Any]:
        layer_type_counts: dict[str, int] = {}
        for layer in self.layers:
            layer_type_counts[layer.layer_type] = layer_type_counts.get(layer.layer_type, 0) + 1

        return {
            "model_path": str(self.model_path),
            "text_only": self.text_only,
            "has_vision_tower": self.config_view.has_vision,
            "nlayer": self.meta.nlayer,
            "hidden_size": self.meta.hs,
            "num_attention_heads": self.meta.nh,
            "num_key_value_heads": self.meta.nkvh,
            "head_dim": self.meta.dh,
            "intermediate_size": self.meta.di,
            "vocab_size": self.meta.voc,
            "max_position_embeddings": self.meta.maxseq,
            "tie_word_embeddings": self.meta.tie_word_embeddings,
            "layer_type_counts": layer_type_counts,
            "shards": list(self.weight_index.shard_files),
            "weight_total_size": self.weight_index.total_size,
            "language_model_layer_weights": self.weight_index.language_model_weight_count,
            "language_model_global_weights": self.weight_index.language_model_global_weight_count,
            "vision_weights": self.weight_index.vision_weight_count,
            "mtp_weights": self.weight_index.mtp_weight_count,
        }

    def _load_shard(self, shard_name: str) -> dict[str, torch.Tensor]:
        shard = self._shard_cache.get(shard_name)
        if shard is None:
            shard_path = self.model_path / shard_name
            shard = safetensors_load_file(str(shard_path))
            self._shard_cache[shard_name] = shard
        return shard

    def _load_host_weight(self, key: str) -> torch.Tensor:
        tensor = self._host_weight_cache.get(key)
        if tensor is not None:
            return tensor

        shard_name = self.weight_index.weight_map.get(key)
        if shard_name is None:
            raise KeyError(f"Weight `{key}` is not present in {self.model_path}")
        shard = self._load_shard(shard_name)
        tensor = shard[key].detach().cpu().contiguous()
        self._host_weight_cache[key] = tensor
        return tensor

    def _to_wginfer_tensor(self, host_tensor: torch.Tensor, dtype: DataType | None = None) -> Tensor:
        host_tensor = host_tensor.detach().cpu().contiguous()
        if dtype is None:
            if host_tensor.dtype == torch.int64:
                dtype = DataType.I64
            else:
                if host_tensor.dtype != torch.bfloat16:
                    host_tensor = host_tensor.to(torch.bfloat16)
                dtype = DataType.BF16
        elif dtype == DataType.BF16 and host_tensor.dtype != torch.bfloat16:
            host_tensor = host_tensor.to(torch.bfloat16)
        elif dtype == DataType.F32 and host_tensor.dtype != torch.float32:
            host_tensor = host_tensor.to(torch.float32)
        elif dtype == DataType.F16 and host_tensor.dtype != torch.float16:
            host_tensor = host_tensor.to(torch.float16)

        tensor = Tensor(shape=list(host_tensor.shape), dtype=dtype, device=self.device)
        tensor.load(host_tensor.data_ptr())
        return tensor

    def _load_weight_tensor(self, key: str, *, add_one: bool = False) -> Tensor:
        cache_key = (key, add_one)
        cached = self._tensor_weight_cache.get(cache_key)
        if cached is not None:
            return cached

        host_tensor = self._load_host_weight(key)
        if add_one:
            host_tensor = host_tensor.to(torch.float32) + 1.0
        tensor = self._to_wginfer_tensor(host_tensor)
        self._tensor_weight_cache[cache_key] = tensor
        return tensor

    def _to_host_torch(self, tensor: Tensor) -> torch.Tensor:
        shape = tuple(tensor.shape())
        if tensor.dtype() == DataType.BF16:
            host = torch.empty(shape, dtype=torch.bfloat16)
        elif tensor.dtype() == DataType.F16:
            host = torch.empty(shape, dtype=torch.float16)
        elif tensor.dtype() == DataType.F32:
            host = torch.empty(shape, dtype=torch.float32)
        elif tensor.dtype() == DataType.I64:
            host = torch.empty(shape, dtype=torch.int64)
        else:
            raise UnsupportedModelConfigError(f"Unsupported tensor dtype in Qwen3.5 host copy: {tensor.dtype()}")

        kind = MemcpyKind.D2D if self.device == DeviceType.CPU else MemcpyKind.D2H
        self._runtime_api.memcpy_sync(host.data_ptr(), tensor.data_ptr(), host.numel() * host.element_size(), kind)
        return host

    def _ensure_tensor(self, shape: tuple[int, ...] | list[int], dtype: DataType | None = None) -> Tensor:
        if dtype is None:
            dtype = self.meta.dtype
        return Tensor(shape=list(shape), dtype=dtype, device=self.device)

    def reset_cache(self):
        self._cache_length = 0
        self._linear_qkv_history: dict[int, torch.Tensor] = {}
        self._linear_recurrent_state: dict[int, Tensor] = {}
        self._full_k_cache: dict[int, Tensor] = {}
        self._full_v_cache: dict[int, Tensor] = {}

    def _cache_memcpy_kind(self) -> MemcpyKind:
        return MemcpyKind.D2D

    @staticmethod
    def _dtype_nbytes(dtype: DataType) -> int:
        if dtype == DataType.F32:
            return 4
        if dtype in {DataType.F16, DataType.BF16}:
            return 2
        if dtype == DataType.I64:
            return 8
        raise UnsupportedModelConfigError(f"Unsupported dtype for cache sizing: {dtype}")

    def _append_to_cache(self, cache: Tensor, values: Tensor, start_pos: int):
        value_shape = values.shape()
        row_elems = 1
        for dim in value_shape[1:]:
            row_elems *= int(dim)
        row_bytes = row_elems * self._dtype_nbytes(values.dtype())
        dst = cache.data_ptr() + start_pos * row_bytes
        self._runtime_api.memcpy_sync(
            dst,
            values.data_ptr(),
            values.numel() * self._dtype_nbytes(values.dtype()),
            self._cache_memcpy_kind(),
        )

    @staticmethod
    def _sample_from_logits_torch(logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> int:
        logits = logits.to(torch.float32)
        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        if top_k <= 0 or top_k > logits.numel():
            top_k = int(logits.numel())
        if top_p <= 0.0 or top_p > 1.0:
            top_p = 1.0

        if top_k == 1 and top_p >= 1.0 and abs(temperature - 1.0) < 1e-6:
            return int(torch.argmax(logits).item())

        scaled = logits / float(temperature)
        values, indices = torch.topk(scaled, k=top_k)
        probs = torch.softmax(values, dim=-1)

        if top_p < 1.0:
            cumulative = torch.cumsum(probs, dim=-1)
            keep = cumulative <= top_p
            if keep.numel() > 0:
                keep[0] = True
            probs = torch.where(keep, probs, torch.zeros_like(probs))
            probs = probs / probs.sum()

        sampled = torch.multinomial(probs, num_samples=1)
        return int(indices[sampled.item()].item())

    @staticmethod
    def _host_rms_norm_with_delta_weight(x: torch.Tensor, weight_delta: torch.Tensor, eps: float) -> torch.Tensor:
        x_f = x.to(torch.float32)
        weight = weight_delta.to(torch.float32) + 1.0
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        return (x_f * torch.rsqrt(variance + eps) * weight).to(x.dtype)

    @staticmethod
    def _apply_partial_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float, partial_rotary_factor: float):
        rotary_dim = int(q.shape[-1] * partial_rotary_factor)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        if rotary_dim <= 0:
            return q, k

        positions = position_ids.to(torch.float32).unsqueeze(1)
        i = torch.arange(0, rotary_dim // 2, dtype=torch.float32, device=q.device)
        freqs = positions / (theta ** (2 * i / rotary_dim))
        sin = freqs.sin().unsqueeze(1)
        cos = freqs.cos().unsqueeze(1)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x_rot = x[..., :rotary_dim]
            x_pass = x[..., rotary_dim:]
            x_a = x_rot[..., : rotary_dim // 2]
            x_b = x_rot[..., rotary_dim // 2 :]
            out_a = x_a * cos - x_b * sin
            out_b = x_b * cos + x_a * sin
            return torch.cat([out_a, out_b, x_pass], dim=-1)

        return rotate(q), rotate(k)

    def _prepare_full_attention_from_host(
        self,
        q_proj_host: torch.Tensor,
        k_proj_host: torch.Tensor,
        v_proj_host: torch.Tensor,
        layer_idx: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seqlen = q_proj_host.shape[0]
        q_host, gate_host = torch.chunk(q_proj_host.view(seqlen, self.meta.nh, self.meta.dh * 2), 2, dim=-1)
        k_host = k_proj_host.view(seqlen, self.meta.nkvh, self.meta.dh)
        v_host = v_proj_host.view(seqlen, self.meta.nkvh, self.meta.dh)

        q_norm_weight = self._load_host_weight(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight")
        k_norm_weight = self._load_host_weight(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight")
        q_host = self._host_rms_norm_with_delta_weight(q_host, q_norm_weight, self.meta.epsilon)
        k_host = self._host_rms_norm_with_delta_weight(k_host, k_norm_weight, self.meta.epsilon)

        position_ids = torch.arange(start_pos, start_pos + seqlen, dtype=torch.int64)
        q_host, k_host = self._apply_partial_rope(
            q_host,
            k_host,
            position_ids,
            self.meta.theta,
            self.meta.partial_rotary_factor,
        )
        gate_flat = gate_host.reshape(seqlen, self.meta.nh * self.meta.dh)
        return q_host, k_host, v_host, gate_flat

    def _host_prepare_linear_inputs(
        self,
        mixed_qkv: Tensor,
        z_flat: Tensor,
        a_flat: Tensor,
        b_flat: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mixed_qkv_host = self._to_host_torch(mixed_qkv).to(torch.float32)
        z_host = self._to_host_torch(z_flat).to(torch.float32)
        a = self._to_host_torch(a_flat).to(torch.float32)
        b = self._to_host_torch(b_flat).to(torch.float32)
        return self._prepare_linear_inputs_from_host(mixed_qkv_host, z_host, a, b, layer_idx)

    def _prepare_linear_inputs_from_host(
        self,
        mixed_qkv_host: torch.Tensor,
        z_host: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        key_dim = self.meta.linear_num_key_heads * self.meta.linear_key_head_dim
        value_dim = self.meta.linear_num_value_heads * self.meta.linear_value_head_dim

        q = mixed_qkv_host[:, :key_dim].view(-1, self.meta.linear_num_key_heads, self.meta.linear_key_head_dim)
        k = mixed_qkv_host[:, key_dim : key_dim * 2].view(
            -1, self.meta.linear_num_key_heads, self.meta.linear_key_head_dim
        )
        v = mixed_qkv_host[:, key_dim * 2 :].view(
            -1, self.meta.linear_num_value_heads, self.meta.linear_value_head_dim
        )
        z = z_host.view(-1, self.meta.linear_num_value_heads, self.meta.linear_value_head_dim)

        q = torch.nn.functional.normalize(q, p=2.0, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1, eps=1e-6)
        q = q * (self.meta.linear_key_head_dim ** -0.5)

        repeat_factor = self.meta.linear_num_value_heads // self.meta.linear_num_key_heads
        q = q.repeat_interleave(repeat_factor, dim=1)
        k = k.repeat_interleave(repeat_factor, dim=1)

        prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
        a_log = self._load_host_weight(prefix + "A_log").to(torch.float32)
        dt_bias = self._load_host_weight(prefix + "dt_bias").to(torch.float32)
        g = -torch.exp(a_log).unsqueeze(0) * torch.nn.functional.softplus(a + dt_bias.unsqueeze(0))
        beta = torch.sigmoid(b)

        return (
            self._to_wginfer_tensor(q, DataType.F32 if self.meta.dtype == DataType.F32 else self.meta.dtype),
            self._to_wginfer_tensor(k, DataType.F32 if self.meta.dtype == DataType.F32 else self.meta.dtype),
            self._to_wginfer_tensor(v, DataType.F32 if self.meta.dtype == DataType.F32 else self.meta.dtype),
            self._to_wginfer_tensor(
                z.view(-1, self.meta.linear_value_head_dim),
                DataType.F32 if self.meta.dtype == DataType.F32 else self.meta.dtype,
            ),
            self._to_wginfer_tensor(g, DataType.F32 if self.meta.dtype == DataType.F32 else self.meta.dtype),
            self._to_wginfer_tensor(beta, DataType.F32 if self.meta.dtype == DataType.F32 else self.meta.dtype),
        )

    def _host_linear_conv(self, qkv_host: torch.Tensor, layer_idx: int) -> torch.Tensor:
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
        weight = self._load_host_weight(prefix + "conv1d.weight").to(torch.float32)
        seq_len = qkv_host.shape[0]
        y = torch.nn.functional.conv1d(
            qkv_host.transpose(0, 1).unsqueeze(0),
            weight,
            bias=None,
            padding=weight.shape[2] - 1,
            groups=weight.shape[0],
        )
        y = torch.nn.functional.silu(y[:, :, :seq_len])
        return y.squeeze(0).transpose(0, 1).contiguous().to(torch.float32)

    def _forward_linear_attention_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        seqlen = int(x.shape()[0])
        prefix = f"model.language_model.layers.{layer_idx}."

        x_norm = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor(prefix + "input_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )

        key_dim = self.meta.linear_num_key_heads * self.meta.linear_key_head_dim
        value_dim = self.meta.linear_num_value_heads * self.meta.linear_value_head_dim
        qkv_dim = key_dim * 2 + value_dim

        qkv = self._ensure_tensor((seqlen, qkv_dim))
        z = self._ensure_tensor((seqlen, value_dim))
        a = self._ensure_tensor((seqlen, self.meta.linear_num_value_heads))
        b = self._ensure_tensor((seqlen, self.meta.linear_num_value_heads))
        Ops.linear(qkv, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_qkv.weight"), None)
        Ops.linear(z, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_z.weight"), None)
        Ops.linear(a, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_a.weight"), None)
        Ops.linear(b, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_b.weight"), None)

        mixed_qkv = self._ensure_tensor((seqlen, qkv_dim))
        Ops.causal_conv1d(mixed_qkv, qkv, self._load_weight_tensor(prefix + "linear_attn.conv1d.weight"))

        q_ready, k_ready, v_ready, z_ready, g_ready, beta_ready = self._host_prepare_linear_inputs(
            mixed_qkv, z, a, b, layer_idx
        )

        if layer_idx == 0:
            self._debug_linear_layer_data = {
                "x_norm": self._to_host_torch(x_norm).to(torch.float32),
                "qkv_proj": self._to_host_torch(qkv).to(torch.float32),
                "z_proj": self._to_host_torch(z).to(torch.float32),
                "a_proj": self._to_host_torch(a).to(torch.float32),
                "b_proj": self._to_host_torch(b).to(torch.float32),
                "mixed_qkv": self._to_host_torch(mixed_qkv).to(torch.float32),
                "q_ready": self._to_host_torch(q_ready).to(torch.float32),
                "k_ready": self._to_host_torch(k_ready).to(torch.float32),
                "v_ready": self._to_host_torch(v_ready).to(torch.float32),
                "z_ready": self._to_host_torch(z_ready).to(torch.float32),
                "g_ready": self._to_host_torch(g_ready).to(torch.float32),
                "beta_ready": self._to_host_torch(beta_ready).to(torch.float32),
            }

        core_attn = self._ensure_tensor((seqlen, self.meta.linear_num_value_heads, self.meta.linear_value_head_dim))
        final_state = self._ensure_tensor(
            (self.meta.linear_num_value_heads, self.meta.linear_key_head_dim, self.meta.linear_value_head_dim)
        )
        Ops.linear_attention(core_attn, q_ready, k_ready, v_ready, g_ready, beta_ready, None, final_state)

        if layer_idx == 0 and self._debug_linear_layer_data is not None:
            self._debug_linear_layer_data["core_attn"] = self._to_host_torch(core_attn).to(torch.float32)

        core_attn_2d = core_attn.view(seqlen * self.meta.linear_num_value_heads, self.meta.linear_value_head_dim)
        gated = self._ensure_tensor((seqlen * self.meta.linear_num_value_heads, self.meta.linear_value_head_dim))
        Ops.gated_rms_norm(
            gated,
            core_attn_2d,
            z_ready,
            self._load_weight_tensor(prefix + "linear_attn.norm.weight"),
            self.meta.epsilon,
        )

        if layer_idx == 0 and self._debug_linear_layer_data is not None:
            self._debug_linear_layer_data["gated"] = self._to_host_torch(gated).to(torch.float32)

        gated_flat = gated.view(seqlen, value_dim)
        attn_proj = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(attn_proj, gated_flat, self._load_weight_tensor(prefix + "linear_attn.out_proj.weight"), None)

        x_attn = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_attn, x, attn_proj)

        if layer_idx == 0 and self._debug_linear_layer_data is not None:
            self._debug_linear_layer_data["attn_proj"] = self._to_host_torch(attn_proj).to(torch.float32)
            self._debug_linear_layer_data["x_attn"] = self._to_host_torch(x_attn).to(torch.float32)

        x_post = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_post,
            x_attn,
            self._load_weight_tensor(prefix + "post_attention_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )
        gate_proj = self._ensure_tensor((seqlen, self.meta.di))
        up_proj = self._ensure_tensor((seqlen, self.meta.di))
        Ops.linear(gate_proj, x_post, self._load_weight_tensor(prefix + "mlp.gate_proj.weight"), None)
        Ops.linear(up_proj, x_post, self._load_weight_tensor(prefix + "mlp.up_proj.weight"), None)

        swiglu = self._ensure_tensor((seqlen, self.meta.di))
        Ops.swiglu(swiglu, gate_proj, up_proj)
        mlp_out = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(mlp_out, swiglu, self._load_weight_tensor(prefix + "mlp.down_proj.weight"), None)

        x_mlp = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_mlp, x_attn, mlp_out)
        return x_mlp

    def _prefill_linear_attention_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        seqlen = int(x.shape()[0])
        prefix = f"model.language_model.layers.{layer_idx}."

        x_norm = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor(prefix + "input_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )

        key_dim = self.meta.linear_num_key_heads * self.meta.linear_key_head_dim
        value_dim = self.meta.linear_num_value_heads * self.meta.linear_value_head_dim
        qkv_dim = key_dim * 2 + value_dim

        qkv = self._ensure_tensor((seqlen, qkv_dim))
        z = self._ensure_tensor((seqlen, value_dim))
        a = self._ensure_tensor((seqlen, self.meta.linear_num_value_heads))
        b = self._ensure_tensor((seqlen, self.meta.linear_num_value_heads))
        Ops.linear(qkv, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_qkv.weight"), None)
        Ops.linear(z, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_z.weight"), None)
        Ops.linear(a, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_a.weight"), None)
        Ops.linear(b, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_b.weight"), None)

        qkv_host = self._to_host_torch(qkv).to(torch.float32)
        z_host = self._to_host_torch(z).to(torch.float32)
        a_host = self._to_host_torch(a).to(torch.float32)
        b_host = self._to_host_torch(b).to(torch.float32)
        mixed_qkv_host = self._host_linear_conv(qkv_host, layer_idx)
        q_ready, k_ready, v_ready, z_ready, g_ready, beta_ready = self._prepare_linear_inputs_from_host(
            mixed_qkv_host,
            z_host,
            a_host,
            b_host,
            layer_idx,
        )

        core_attn = self._ensure_tensor((seqlen, self.meta.linear_num_value_heads, self.meta.linear_value_head_dim))
        final_state = self._ensure_tensor(
            (self.meta.linear_num_value_heads, self.meta.linear_key_head_dim, self.meta.linear_value_head_dim)
        )
        Ops.linear_attention(core_attn, q_ready, k_ready, v_ready, g_ready, beta_ready, None, final_state)

        keep = self.meta.linear_conv_kernel_dim - 1
        self._linear_qkv_history[layer_idx] = qkv_host[-keep:].clone() if keep > 0 else qkv_host[:0].clone()
        self._linear_recurrent_state[layer_idx] = final_state

        core_attn_2d = core_attn.view(seqlen * self.meta.linear_num_value_heads, self.meta.linear_value_head_dim)
        gated = self._ensure_tensor((seqlen * self.meta.linear_num_value_heads, self.meta.linear_value_head_dim))
        Ops.gated_rms_norm(
            gated,
            core_attn_2d,
            z_ready,
            self._load_weight_tensor(prefix + "linear_attn.norm.weight"),
            self.meta.epsilon,
        )

        gated_flat = gated.view(seqlen, value_dim)
        attn_proj = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(attn_proj, gated_flat, self._load_weight_tensor(prefix + "linear_attn.out_proj.weight"), None)
        x_attn = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_attn, x, attn_proj)

        x_post = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_post,
            x_attn,
            self._load_weight_tensor(prefix + "post_attention_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )
        gate_proj = self._ensure_tensor((seqlen, self.meta.di))
        up_proj = self._ensure_tensor((seqlen, self.meta.di))
        Ops.linear(gate_proj, x_post, self._load_weight_tensor(prefix + "mlp.gate_proj.weight"), None)
        Ops.linear(up_proj, x_post, self._load_weight_tensor(prefix + "mlp.up_proj.weight"), None)
        swiglu = self._ensure_tensor((seqlen, self.meta.di))
        Ops.swiglu(swiglu, gate_proj, up_proj)
        mlp_out = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(mlp_out, swiglu, self._load_weight_tensor(prefix + "mlp.down_proj.weight"), None)
        x_mlp = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_mlp, x_attn, mlp_out)
        return x_mlp

    def _decode_linear_attention_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        prefix = f"model.language_model.layers.{layer_idx}."
        x_norm = self._ensure_tensor((1, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor(prefix + "input_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )

        key_dim = self.meta.linear_num_key_heads * self.meta.linear_key_head_dim
        value_dim = self.meta.linear_num_value_heads * self.meta.linear_value_head_dim
        qkv_dim = key_dim * 2 + value_dim

        qkv = self._ensure_tensor((1, qkv_dim))
        z = self._ensure_tensor((1, value_dim))
        a = self._ensure_tensor((1, self.meta.linear_num_value_heads))
        b = self._ensure_tensor((1, self.meta.linear_num_value_heads))
        Ops.linear(qkv, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_qkv.weight"), None)
        Ops.linear(z, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_z.weight"), None)
        Ops.linear(a, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_a.weight"), None)
        Ops.linear(b, x_norm, self._load_weight_tensor(prefix + "linear_attn.in_proj_b.weight"), None)

        qkv_host = self._to_host_torch(qkv).to(torch.float32)
        z_host = self._to_host_torch(z).to(torch.float32)
        a_host = self._to_host_torch(a).to(torch.float32)
        b_host = self._to_host_torch(b).to(torch.float32)
        history = self._linear_qkv_history.get(layer_idx)
        if history is None:
            history = torch.empty((0, qkv_host.shape[1]), dtype=torch.float32)
        combined_qkv = torch.cat([history, qkv_host], dim=0)
        mixed_qkv_last = self._host_linear_conv(combined_qkv, layer_idx)[-1:].contiguous()
        q_ready, k_ready, v_ready, z_ready, g_ready, beta_ready = self._prepare_linear_inputs_from_host(
            mixed_qkv_last,
            z_host,
            a_host,
            b_host,
            layer_idx,
        )

        initial_state = self._linear_recurrent_state[layer_idx]
        final_state = self._ensure_tensor(
            (self.meta.linear_num_value_heads, self.meta.linear_key_head_dim, self.meta.linear_value_head_dim)
        )
        core_attn = self._ensure_tensor((1, self.meta.linear_num_value_heads, self.meta.linear_value_head_dim))
        Ops.linear_attention(core_attn, q_ready, k_ready, v_ready, g_ready, beta_ready, initial_state, final_state)

        keep = self.meta.linear_conv_kernel_dim - 1
        self._linear_qkv_history[layer_idx] = combined_qkv[-keep:].clone() if keep > 0 else combined_qkv[:0].clone()
        self._linear_recurrent_state[layer_idx] = final_state

        core_attn_2d = core_attn.view(self.meta.linear_num_value_heads, self.meta.linear_value_head_dim)
        gated = self._ensure_tensor((self.meta.linear_num_value_heads, self.meta.linear_value_head_dim))
        Ops.gated_rms_norm(
            gated,
            core_attn_2d,
            z_ready,
            self._load_weight_tensor(prefix + "linear_attn.norm.weight"),
            self.meta.epsilon,
        )

        gated_flat = gated.view(1, value_dim)
        attn_proj = self._ensure_tensor((1, self.meta.hs))
        Ops.linear(attn_proj, gated_flat, self._load_weight_tensor(prefix + "linear_attn.out_proj.weight"), None)
        x_attn = self._ensure_tensor((1, self.meta.hs))
        Ops.add(x_attn, x, attn_proj)

        x_post = self._ensure_tensor((1, self.meta.hs))
        Ops.rms_norm(
            x_post,
            x_attn,
            self._load_weight_tensor(prefix + "post_attention_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )
        gate_proj = self._ensure_tensor((1, self.meta.di))
        up_proj = self._ensure_tensor((1, self.meta.di))
        Ops.linear(gate_proj, x_post, self._load_weight_tensor(prefix + "mlp.gate_proj.weight"), None)
        Ops.linear(up_proj, x_post, self._load_weight_tensor(prefix + "mlp.up_proj.weight"), None)
        swiglu = self._ensure_tensor((1, self.meta.di))
        Ops.swiglu(swiglu, gate_proj, up_proj)
        mlp_out = self._ensure_tensor((1, self.meta.hs))
        Ops.linear(mlp_out, swiglu, self._load_weight_tensor(prefix + "mlp.down_proj.weight"), None)
        x_mlp = self._ensure_tensor((1, self.meta.hs))
        Ops.add(x_mlp, x_attn, mlp_out)
        return x_mlp

    def _forward_full_attention_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        seqlen = int(x.shape()[0])
        prefix = f"model.language_model.layers.{layer_idx}."

        x_norm = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor(prefix + "input_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )

        q_proj = self._ensure_tensor((seqlen, self.meta.nh * self.meta.dh * 2))
        k_proj = self._ensure_tensor((seqlen, self.meta.nkvh * self.meta.dh))
        v_proj = self._ensure_tensor((seqlen, self.meta.nkvh * self.meta.dh))
        Ops.linear(q_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.q_proj.weight"), None)
        Ops.linear(k_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.k_proj.weight"), None)
        Ops.linear(v_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.v_proj.weight"), None)

        q_proj_host = self._to_host_torch(q_proj).to(torch.float32)
        k_proj_host = self._to_host_torch(k_proj).to(torch.float32)
        v_proj_host = self._to_host_torch(v_proj).to(torch.float32)

        q_host, gate_host = torch.chunk(q_proj_host.view(seqlen, self.meta.nh, self.meta.dh * 2), 2, dim=-1)
        k_host = k_proj_host.view(seqlen, self.meta.nkvh, self.meta.dh)
        v_host = v_proj_host.view(seqlen, self.meta.nkvh, self.meta.dh)

        q_norm_weight = self._load_host_weight(prefix + "self_attn.q_norm.weight")
        k_norm_weight = self._load_host_weight(prefix + "self_attn.k_norm.weight")
        q_host = self._host_rms_norm_with_delta_weight(q_host, q_norm_weight, self.meta.epsilon)
        k_host = self._host_rms_norm_with_delta_weight(k_host, k_norm_weight, self.meta.epsilon)

        position_ids = torch.arange(seqlen, dtype=torch.int64)
        q_host, k_host = self._apply_partial_rope(
            q_host,
            k_host,
            position_ids,
            self.meta.theta,
            self.meta.partial_rotary_factor,
        )

        q_ready = self._to_wginfer_tensor(q_host, self.meta.dtype)
        k_ready = self._to_wginfer_tensor(k_host, self.meta.dtype)
        v_ready = self._to_wginfer_tensor(v_host, self.meta.dtype)
        gate_ready = self._to_wginfer_tensor(gate_host.reshape(seqlen, self.meta.nh * self.meta.dh), self.meta.dtype)

        attn_out = self._ensure_tensor((seqlen, self.meta.nh, self.meta.dh))
        Ops.self_attention(attn_out, q_ready, k_ready, v_ready, self.meta.dh ** -0.5)
        attn_flat = attn_out.view(seqlen, self.meta.nh * self.meta.dh)

        attn_flat_host = self._to_host_torch(attn_flat).to(torch.float32)
        gate_host_flat = self._to_host_torch(gate_ready).to(torch.float32)
        gated_attn_host = attn_flat_host * torch.sigmoid(gate_host_flat)
        gated_attn = self._to_wginfer_tensor(gated_attn_host, self.meta.dtype)

        attn_proj = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(attn_proj, gated_attn, self._load_weight_tensor(prefix + "self_attn.o_proj.weight"), None)

        x_attn = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_attn, x, attn_proj)

        x_post = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_post,
            x_attn,
            self._load_weight_tensor(prefix + "post_attention_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )
        gate_proj = self._ensure_tensor((seqlen, self.meta.di))
        up_proj = self._ensure_tensor((seqlen, self.meta.di))
        Ops.linear(gate_proj, x_post, self._load_weight_tensor(prefix + "mlp.gate_proj.weight"), None)
        Ops.linear(up_proj, x_post, self._load_weight_tensor(prefix + "mlp.up_proj.weight"), None)

        swiglu = self._ensure_tensor((seqlen, self.meta.di))
        Ops.swiglu(swiglu, gate_proj, up_proj)
        mlp_out = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(mlp_out, swiglu, self._load_weight_tensor(prefix + "mlp.down_proj.weight"), None)

        x_mlp = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_mlp, x_attn, mlp_out)
        return x_mlp

    def _prefill_full_attention_layer(self, x: Tensor, layer_idx: int, start_pos: int) -> Tensor:
        seqlen = int(x.shape()[0])
        prefix = f"model.language_model.layers.{layer_idx}."

        x_norm = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor(prefix + "input_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )

        q_proj = self._ensure_tensor((seqlen, self.meta.nh * self.meta.dh * 2))
        k_proj = self._ensure_tensor((seqlen, self.meta.nkvh * self.meta.dh))
        v_proj = self._ensure_tensor((seqlen, self.meta.nkvh * self.meta.dh))
        Ops.linear(q_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.q_proj.weight"), None)
        Ops.linear(k_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.k_proj.weight"), None)
        Ops.linear(v_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.v_proj.weight"), None)

        q_proj_host = self._to_host_torch(q_proj).to(torch.float32)
        k_proj_host = self._to_host_torch(k_proj).to(torch.float32)
        v_proj_host = self._to_host_torch(v_proj).to(torch.float32)
        q_host, k_host, v_host, gate_flat = self._prepare_full_attention_from_host(
            q_proj_host,
            k_proj_host,
            v_proj_host,
            layer_idx,
            start_pos,
        )
        q_ready = self._to_wginfer_tensor(q_host, self.meta.dtype)
        k_ready = self._to_wginfer_tensor(k_host, self.meta.dtype)
        v_ready = self._to_wginfer_tensor(v_host, self.meta.dtype)
        if layer_idx not in self._full_k_cache:
            self._full_k_cache[layer_idx] = self._ensure_tensor((self.meta.maxseq, self.meta.nkvh, self.meta.dh))
            self._full_v_cache[layer_idx] = self._ensure_tensor((self.meta.maxseq, self.meta.nkvh, self.meta.dh))
        self._append_to_cache(self._full_k_cache[layer_idx], k_ready, start_pos)
        self._append_to_cache(self._full_v_cache[layer_idx], v_ready, start_pos)

        attn_out = self._ensure_tensor((seqlen, self.meta.nh, self.meta.dh))
        Ops.self_attention(attn_out, q_ready, k_ready, v_ready, self.meta.dh ** -0.5)
        attn_flat = attn_out.view(seqlen, self.meta.nh * self.meta.dh)
        attn_flat_host = self._to_host_torch(attn_flat).to(torch.float32)
        gated_attn_host = attn_flat_host * torch.sigmoid(gate_flat)
        gated_attn = self._to_wginfer_tensor(gated_attn_host, self.meta.dtype)

        attn_proj = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(attn_proj, gated_attn, self._load_weight_tensor(prefix + "self_attn.o_proj.weight"), None)
        x_attn = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_attn, x, attn_proj)

        x_post = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_post,
            x_attn,
            self._load_weight_tensor(prefix + "post_attention_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )
        gate_proj = self._ensure_tensor((seqlen, self.meta.di))
        up_proj = self._ensure_tensor((seqlen, self.meta.di))
        Ops.linear(gate_proj, x_post, self._load_weight_tensor(prefix + "mlp.gate_proj.weight"), None)
        Ops.linear(up_proj, x_post, self._load_weight_tensor(prefix + "mlp.up_proj.weight"), None)
        swiglu = self._ensure_tensor((seqlen, self.meta.di))
        Ops.swiglu(swiglu, gate_proj, up_proj)
        mlp_out = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.linear(mlp_out, swiglu, self._load_weight_tensor(prefix + "mlp.down_proj.weight"), None)
        x_mlp = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.add(x_mlp, x_attn, mlp_out)
        return x_mlp

    def _decode_full_attention_layer(self, x: Tensor, layer_idx: int, start_pos: int) -> Tensor:
        prefix = f"model.language_model.layers.{layer_idx}."

        x_norm = self._ensure_tensor((1, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor(prefix + "input_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )

        q_proj = self._ensure_tensor((1, self.meta.nh * self.meta.dh * 2))
        k_proj = self._ensure_tensor((1, self.meta.nkvh * self.meta.dh))
        v_proj = self._ensure_tensor((1, self.meta.nkvh * self.meta.dh))
        Ops.linear(q_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.q_proj.weight"), None)
        Ops.linear(k_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.k_proj.weight"), None)
        Ops.linear(v_proj, x_norm, self._load_weight_tensor(prefix + "self_attn.v_proj.weight"), None)

        q_proj_host = self._to_host_torch(q_proj).to(torch.float32)
        k_proj_host = self._to_host_torch(k_proj).to(torch.float32)
        v_proj_host = self._to_host_torch(v_proj).to(torch.float32)
        q_host, k_host, v_host, gate_flat = self._prepare_full_attention_from_host(
            q_proj_host,
            k_proj_host,
            v_proj_host,
            layer_idx,
            start_pos,
        )
        q_ready = self._to_wginfer_tensor(q_host, self.meta.dtype)
        k_new = self._to_wginfer_tensor(k_host, self.meta.dtype)
        v_new = self._to_wginfer_tensor(v_host, self.meta.dtype)
        self._append_to_cache(self._full_k_cache[layer_idx], k_new, start_pos)
        self._append_to_cache(self._full_v_cache[layer_idx], v_new, start_pos)
        k_ready = self._full_k_cache[layer_idx].slice(0, 0, start_pos + 1)
        v_ready = self._full_v_cache[layer_idx].slice(0, 0, start_pos + 1)

        attn_out = self._ensure_tensor((1, self.meta.nh, self.meta.dh))
        Ops.self_attention(attn_out, q_ready, k_ready, v_ready, self.meta.dh ** -0.5)
        attn_flat = attn_out.view(1, self.meta.nh * self.meta.dh)
        attn_flat_host = self._to_host_torch(attn_flat).to(torch.float32)
        gated_attn_host = attn_flat_host * torch.sigmoid(gate_flat)
        gated_attn = self._to_wginfer_tensor(gated_attn_host, self.meta.dtype)

        attn_proj = self._ensure_tensor((1, self.meta.hs))
        Ops.linear(attn_proj, gated_attn, self._load_weight_tensor(prefix + "self_attn.o_proj.weight"), None)
        x_attn = self._ensure_tensor((1, self.meta.hs))
        Ops.add(x_attn, x, attn_proj)

        x_post = self._ensure_tensor((1, self.meta.hs))
        Ops.rms_norm(
            x_post,
            x_attn,
            self._load_weight_tensor(prefix + "post_attention_layernorm.weight", add_one=True),
            self.meta.epsilon,
        )
        gate_proj = self._ensure_tensor((1, self.meta.di))
        up_proj = self._ensure_tensor((1, self.meta.di))
        Ops.linear(gate_proj, x_post, self._load_weight_tensor(prefix + "mlp.gate_proj.weight"), None)
        Ops.linear(up_proj, x_post, self._load_weight_tensor(prefix + "mlp.up_proj.weight"), None)
        swiglu = self._ensure_tensor((1, self.meta.di))
        Ops.swiglu(swiglu, gate_proj, up_proj)
        mlp_out = self._ensure_tensor((1, self.meta.hs))
        Ops.linear(mlp_out, swiglu, self._load_weight_tensor(prefix + "mlp.down_proj.weight"), None)
        x_mlp = self._ensure_tensor((1, self.meta.hs))
        Ops.add(x_mlp, x_attn, mlp_out)
        return x_mlp

    def forward_prefix(self, token_ids: list[int] | tuple[int, ...], max_layers: int = 1) -> Tensor:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        max_layers = max(int(max_layers), 0)
        max_layers = min(max_layers, self.meta.nlayer)

        input_ids = self._to_wginfer_tensor(torch.tensor(list(token_ids), dtype=torch.int64), DataType.I64)
        x = self._ensure_tensor((len(token_ids), self.meta.hs))
        Ops.embedding(x, input_ids, self._load_weight_tensor("model.language_model.embed_tokens.weight"))

        for layer in self.layers[:max_layers]:
            if layer.layer_type == "linear_attention":
                x = self._forward_linear_attention_layer(x, layer.index)
            elif layer.layer_type == "full_attention":
                x = self._forward_full_attention_layer(x, layer.index)
            else:
                raise NotImplementedError(f"Unsupported Qwen3.5 layer type: {layer.layer_type}")

        return x

    def forward_prefix_host(self, token_ids: list[int] | tuple[int, ...], max_layers: int = 1) -> torch.Tensor:
        return self._to_host_torch(self.forward_prefix(token_ids, max_layers))

    def forward_logits(self, token_ids: list[int] | tuple[int, ...], max_layers: int | None = None) -> Tensor:
        if max_layers is None:
            max_layers = self.meta.nlayer
        x = self.forward_prefix(token_ids, max_layers=max_layers)
        x_norm = self._ensure_tensor((len(token_ids), self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor("model.language_model.norm.weight", add_one=True),
            self.meta.epsilon,
        )
        logits = self._ensure_tensor((len(token_ids), self.meta.voc))
        Ops.linear(logits, x_norm, self._load_weight_tensor("model.language_model.embed_tokens.weight"), None)
        return logits

    def forward_logits_host(self, token_ids: list[int] | tuple[int, ...], max_layers: int | None = None) -> torch.Tensor:
        return self._to_host_torch(self.forward_logits(token_ids, max_layers=max_layers))

    def _final_logits_from_hidden(self, x: Tensor, seqlen: int) -> Tensor:
        x_norm = self._ensure_tensor((seqlen, self.meta.hs))
        Ops.rms_norm(
            x_norm,
            x,
            self._load_weight_tensor("model.language_model.norm.weight", add_one=True),
            self.meta.epsilon,
        )
        logits = self._ensure_tensor((seqlen, self.meta.voc))
        Ops.linear(logits, x_norm, self._load_weight_tensor("model.language_model.embed_tokens.weight"), None)
        return logits

    def _prefill_logits_host(self, token_ids: list[int] | tuple[int, ...], max_layers: int | None = None) -> torch.Tensor:
        return self._to_host_torch(self._prefill_logits(token_ids, max_layers=max_layers))

    def _prefill_logits(self, token_ids: list[int] | tuple[int, ...], max_layers: int | None = None) -> Tensor:
        if max_layers is None:
            max_layers = self.meta.nlayer
        input_ids = self._to_wginfer_tensor(torch.tensor(list(token_ids), dtype=torch.int64), DataType.I64)
        x = self._ensure_tensor((len(token_ids), self.meta.hs))
        Ops.embedding(x, input_ids, self._load_weight_tensor("model.language_model.embed_tokens.weight"))
        for layer in self.layers[:max_layers]:
            if layer.layer_type == "linear_attention":
                x = self._prefill_linear_attention_layer(x, layer.index)
            elif layer.layer_type == "full_attention":
                x = self._prefill_full_attention_layer(x, layer.index, start_pos=0)
            else:
                raise NotImplementedError(f"Unsupported Qwen3.5 layer type: {layer.layer_type}")
        self._cache_length = len(token_ids)
        return self._final_logits_from_hidden(x, len(token_ids))

    def _decode_one_token_logits_host(self, token_id: int, max_layers: int | None = None) -> torch.Tensor:
        return self._to_host_torch(self._decode_one_token_logits(token_id, max_layers=max_layers))

    def _decode_one_token_logits(self, token_id: int, max_layers: int | None = None) -> Tensor:
        if max_layers is None:
            max_layers = self.meta.nlayer
        input_ids = self._to_wginfer_tensor(torch.tensor([token_id], dtype=torch.int64), DataType.I64)
        x = self._ensure_tensor((1, self.meta.hs))
        Ops.embedding(x, input_ids, self._load_weight_tensor("model.language_model.embed_tokens.weight"))
        for layer in self.layers[:max_layers]:
            if layer.layer_type == "linear_attention":
                x = self._decode_linear_attention_layer(x, layer.index)
            elif layer.layer_type == "full_attention":
                x = self._decode_full_attention_layer(x, layer.index, start_pos=self._cache_length)
            else:
                raise NotImplementedError(f"Unsupported Qwen3.5 layer type: {layer.layer_type}")
        self._cache_length += 1
        return self._final_logits_from_hidden(x, 1)

    def _greedy_next_token_from_logits(self, logits: Tensor) -> int:
        last_logits = logits.slice(0, logits.shape()[0] - 1, logits.shape()[0]).view(self.meta.voc)
        if self._argmax_idx is None:
            self._argmax_idx = self._ensure_tensor((1,), DataType.I64)
        if self._argmax_val is None:
            self._argmax_val = self._ensure_tensor((1,), self.meta.dtype)
        Ops.argmax(self._argmax_idx, self._argmax_val, last_logits)
        host_result = torch.empty((1,), dtype=torch.int64)
        kind = MemcpyKind.D2D if self.device == DeviceType.CPU else MemcpyKind.D2H
        self._runtime_api.memcpy_sync(
            host_result.data_ptr(),
            self._argmax_idx.data_ptr(),
            host_result.numel() * host_result.element_size(),
            kind,
        )
        return int(host_result.item())

    def debug_linear_layer_host(self, token_ids: list[int] | tuple[int, ...]) -> dict[str, torch.Tensor]:
        self._debug_linear_layer_data = None
        self.forward_prefix(token_ids, max_layers=1)
        if self._debug_linear_layer_data is None:
            raise RuntimeError("Linear-attention debug data was not captured")
        return self._debug_linear_layer_data

    def generate(
        self,
        inputs,
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_layers: int | None = None,
    ):
        if self.runtime_model is not None and max_layers is None:
            self._load_runtime_weights()
            output_tokens = list(inputs)
            if not output_tokens:
                return output_tokens

            max_new_tokens = max(int(max_new_tokens), 1)
            self.runtime_model.reset_cache()

            next_token = int(
                self.runtime_model.infer(
                    list(output_tokens),
                    int(top_k),
                    float(top_p),
                    float(temperature),
                )
            )
            output_tokens.append(next_token)
            for _ in range(max_new_tokens - 1):
                if next_token == self.meta.end_token:
                    break
                next_token = int(
                    self.runtime_model.infer(
                        [next_token],
                        int(top_k),
                        float(top_p),
                        float(temperature),
                    )
                )
                output_tokens.append(next_token)
            return output_tokens

        output_tokens = list(inputs)
        if not output_tokens:
            return output_tokens

        max_new_tokens = max(int(max_new_tokens), 1)
        self.reset_cache()
        greedy = top_k == 1 and top_p >= 1.0 and abs(temperature - 1.0) < 1e-6
        logits = self._prefill_logits(output_tokens, max_layers=max_layers)
        if greedy:
            next_token = self._greedy_next_token_from_logits(logits)
        else:
            next_token = self._sample_from_logits_torch(self._to_host_torch(logits)[-1], top_k, top_p, temperature)
        output_tokens.append(next_token)
        if next_token == self.meta.end_token:
            return output_tokens

        for _ in range(max_new_tokens - 1):
            logits = self._decode_one_token_logits(next_token, max_layers=max_layers)
            if greedy:
                next_token = self._greedy_next_token_from_logits(logits)
            else:
                next_token = self._sample_from_logits_torch(self._to_host_torch(logits)[-1], top_k, top_p, temperature)
            output_tokens.append(next_token)
            if next_token == self.meta.end_token:
                break
        return output_tokens

    def generate_stream(
        self,
        inputs,
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_layers: int | None = None,
    ):
        if self.runtime_model is not None and max_layers is None:
            self._load_runtime_weights()
            output_tokens = list(inputs)
            if not output_tokens:
                return

            max_new_tokens = max(int(max_new_tokens), 1)
            self.runtime_model.reset_cache()

            next_token = int(
                self.runtime_model.infer(
                    list(output_tokens),
                    int(top_k),
                    float(top_p),
                    float(temperature),
                )
            )
            output_tokens.append(next_token)
            yield next_token
            for _ in range(max_new_tokens - 1):
                if next_token == self.meta.end_token:
                    break
                next_token = int(
                    self.runtime_model.infer(
                        [next_token],
                        int(top_k),
                        float(top_p),
                        float(temperature),
                    )
                )
                output_tokens.append(next_token)
                yield next_token
            return

        output_tokens = list(inputs)
        if not output_tokens:
            return

        max_new_tokens = max(int(max_new_tokens), 1)
        self.reset_cache()
        greedy = top_k == 1 and top_p >= 1.0 and abs(temperature - 1.0) < 1e-6
        logits = self._prefill_logits(output_tokens, max_layers=max_layers)
        if greedy:
            next_token = self._greedy_next_token_from_logits(logits)
        else:
            next_token = self._sample_from_logits_torch(self._to_host_torch(logits)[-1], top_k, top_p, temperature)
        output_tokens.append(next_token)
        yield next_token
        if next_token == self.meta.end_token:
            return

        for _ in range(max_new_tokens - 1):
            logits = self._decode_one_token_logits(next_token, max_layers=max_layers)
            if greedy:
                next_token = self._greedy_next_token_from_logits(logits)
            else:
                next_token = self._sample_from_logits_torch(self._to_host_torch(logits)[-1], top_k, top_p, temperature)
            output_tokens.append(next_token)
            yield next_token
            if next_token == self.meta.end_token:
                break
