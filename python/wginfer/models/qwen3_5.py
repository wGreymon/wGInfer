from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file as safetensors_load_file

from .._wginfer.core import DataType
from .._wginfer.core import DeviceType
from .._wginfer.core import MemcpyKind
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
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
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
        self._runtime_api = RuntimeAPI(device)
        self._runtime_weights_loaded = False
        self._init_runtime_model()

    def _init_runtime_model(self) -> None:
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

    def _ensure_runtime_ready(self) -> None:
        if self.runtime_model is None:
            raise RuntimeError("Qwen3.5 runtime backend is unavailable; rebuild the C++ extension first.")
        self._load_runtime_weights()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> list[int]:
        self._ensure_runtime_ready()

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

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        self._ensure_runtime_ready()
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
