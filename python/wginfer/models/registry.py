from __future__ import annotations

from pathlib import Path

from .._wginfer.core import DeviceType
from ._config import UnsupportedModelConfigError
from .qwen2 import Qwen2
from .qwen3_5 import Qwen3_5


MODEL_REGISTRY = {
    "qwen2": Qwen2,
    "qwen3_5": Qwen3_5,
}


def supported_model_types() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))


def get_model_class(model_type: str):
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError as exc:
        options = ", ".join(supported_model_types())
        raise UnsupportedModelConfigError(
            f"Unsupported model_type `{model_type}`. Supported values: {options}."
        ) from exc


def create_model(model_type: str, model_path: str | Path, device: DeviceType = DeviceType.CPU):
    model_cls = get_model_class(model_type)
    return model_cls(model_path, device)
