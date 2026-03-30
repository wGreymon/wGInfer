from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsupportedModelConfigError(NotImplementedError):
    pass


@dataclass(frozen=True)
class ModelConfigView:
    raw_config: dict[str, Any]
    text_config: dict[str, Any]
    model_type: str
    text_model_type: str
    architectures: tuple[str, ...]
    has_vision: bool
    layer_types: tuple[str, ...]


def load_model_config(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_config(config: dict[str, Any]) -> ModelConfigView:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config

    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list):
        layer_types = []

    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        architectures = []

    return ModelConfigView(
        raw_config=config,
        text_config=text_config,
        model_type=str(config.get("model_type", "")),
        text_model_type=str(text_config.get("model_type", config.get("model_type", ""))),
        architectures=tuple(str(x) for x in architectures),
        has_vision=isinstance(config.get("vision_config"), dict),
        layer_types=tuple(str(x) for x in layer_types),
    )


def rope_theta_from_text_config(text_config: dict[str, Any], default: float = 1_000_000.0) -> float:
    if "rope_theta" in text_config:
        return float(text_config["rope_theta"])
    rope_parameters = text_config.get("rope_parameters")
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    return float(default)
