from __future__ import annotations

import torch

from typing import Sequence

from .._wginfer.core import DataType
from .._wginfer.core import DeviceType
from .._wginfer.core import MemcpyKind
from ._config import UnsupportedModelConfigError
from .qwen3_5_reference import Qwen3_5 as _ReferenceQwen3_5
from .qwen3_5_reference import Qwen3_5LayerSpec
from .qwen3_5_reference import Qwen3_5TextMeta
from .qwen3_5_reference import Qwen3_5WeightIndex
from .qwen3_5_reference import is_qwen3_5_config
from .qwen3_5_reference import load_qwen3_5_weight_index
from .qwen3_5_reference import parse_qwen3_5_text_meta


class Qwen3_5(_ReferenceQwen3_5):
    """Thin runtime-backed Qwen3.5 wrapper.

    The production inference path is intentionally aligned with Qwen2:
    Python handles config/weight loading, while the C++ backend owns
    prefill/decode/cache/sampling through ``runtime_model.infer``.

    The inherited reference helpers remain available for debugging and
    intermediate-tensor inspection, but they are no longer used by the
    default generate path.
    """

    def _ensure_runtime_ready(self) -> None:
        if self.runtime_model is None:
            raise RuntimeError("Qwen3.5 runtime backend is unavailable; rebuild the C++ extension first.")
        self._load_runtime_weights()

    def _generate_via_runtime(
        self,
        inputs: Sequence[int],
        max_new_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
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

    def runtime_forward_logits(self, token_ids: Sequence[int]):
        self._ensure_runtime_ready()
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        return self.runtime_model.forward_logits(list(token_ids))

    def runtime_forward_logits_host(self, token_ids: Sequence[int]) -> torch.Tensor:
        logits = self.runtime_forward_logits(token_ids)
        shape = tuple(logits.shape())
        if logits.dtype() == DataType.BF16:
            host = torch.empty(shape, dtype=torch.bfloat16)
        elif logits.dtype() == DataType.F16:
            host = torch.empty(shape, dtype=torch.float16)
        elif logits.dtype() == DataType.F32:
            host = torch.empty(shape, dtype=torch.float32)
        else:
            raise UnsupportedModelConfigError(
                f"Unsupported runtime logits dtype for host copy: {logits.dtype()}"
            )
        kind = MemcpyKind.D2D if self.device == DeviceType.CPU else MemcpyKind.D2H
        self._runtime_api.memcpy_sync(
            host.data_ptr(),
            logits.data_ptr(),
            host.numel() * host.element_size(),
            kind,
        )
        return host

    def generate(
        self,
        inputs,
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_layers: int | None = None,
    ):
        if max_layers is not None:
            raise NotImplementedError(
                "Qwen3.5 max_layers debugging remains in qwen3_5_reference; "
                "the default Qwen3_5 path only uses the C++ backend."
            )

        return self._generate_via_runtime(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    def generate_stream(
        self,
        inputs,
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_layers: int | None = None,
    ):
        if max_layers is not None:
            raise NotImplementedError(
                "Qwen3.5 max_layers debugging remains in qwen3_5_reference; "
                "the default Qwen3_5 path only uses the C++ backend."
            )

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
