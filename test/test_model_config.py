import json
import tempfile
from pathlib import Path

from wginfer.core import DeviceType
from wginfer.models._config import UnsupportedModelConfigError
from wginfer.models._config import resolve_model_config
from wginfer.models._config import rope_theta_from_text_config
from wginfer.models.qwen2 import ensure_qwen2_compatible
from wginfer.models.qwen2 import is_qwen2_config
from wginfer.models.qwen3_5 import Qwen3_5
from wginfer.models.qwen3_5 import is_qwen3_5_config
from wginfer.models.registry import create_model
from wginfer.models.registry import get_model_class
from wginfer.models.registry import supported_model_types


def test_qwen2_compatible_config():
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 8960,
        "max_position_embeddings": 32768,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }
    view = resolve_model_config(config)
    assert is_qwen2_config(view) is True
    assert is_qwen3_5_config(view) is False
    ensure_qwen2_compatible(view, "/tmp/qwen2")
    assert rope_theta_from_text_config(view.text_config) == 1000000.0


def test_nested_text_config_and_rope_parameters():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {"rope_theta": 10000000},
        },
        "vision_config": {"hidden_size": 1024},
    }
    view = resolve_model_config(config)
    assert view.model_type == "qwen3_5"
    assert view.text_model_type == "qwen3_5_text"
    assert view.layer_types == ("linear_attention", "full_attention")
    assert view.has_vision is True
    assert is_qwen2_config(view) is False
    assert is_qwen3_5_config(view) is True
    assert rope_theta_from_text_config(view.text_config) == 10000000.0


def test_reject_qwen3_5_for_qwen2_loader():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
        "vision_config": {"hidden_size": 1024},
    }
    view = resolve_model_config(config)
    try:
        ensure_qwen2_compatible(view, "/tmp/Qwen3.5-4B")
    except UnsupportedModelConfigError as exc:
        msg = str(exc)
        assert "vision_config" in msg
        assert "qwen3_5_text" in msg
        assert "layer_types" in msg
    else:
        raise AssertionError("Qwen3.5 config should be rejected by the Qwen2 loader")


def test_real_qwen3_5_config_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {"rope_theta": 10000000},
            },
            "vision_config": {"hidden_size": 1024},
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        loaded = json.loads((root / "config.json").read_text(encoding="utf-8"))
        view = resolve_model_config(loaded)
        assert view.model_type == "qwen3_5"


def test_registry_supports_explicit_model_types():
    assert supported_model_types() == ("qwen2", "qwen3_5")
    assert get_model_class("qwen2").__name__ == "Qwen2"
    assert get_model_class("qwen3_5").__name__ == "Qwen3_5"


def test_create_model_builds_qwen3_5_skeleton():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 2,
                        "layer_types": ["linear_attention", "full_attention"],
                    },
                    "vision_config": {"hidden_size": 1024},
                }
            ),
            encoding="utf-8",
        )
        weight_map = {
            "model.language_model.embed_tokens.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.norm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.input_layernorm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.post_attention_layernorm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.mlp.gate_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.mlp.up_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.mlp.down_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.out_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.norm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.dt_bias": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.A_log": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.0.linear_attn.conv1d.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.input_layernorm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.post_attention_layernorm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.mlp.gate_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.mlp.up_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.mlp.down_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.self_attn.q_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.self_attn.k_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.self_attn.v_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.self_attn.o_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.self_attn.q_norm.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.language_model.layers.1.self_attn.k_norm.weight": "model.safetensors-00001-of-00001.safetensors",
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 1234},
                    "weight_map": weight_map,
                }
            ),
            encoding="utf-8",
        )

        model = create_model("qwen3_5", root, DeviceType.CPU)
        assert isinstance(model, Qwen3_5)
        summary = model.summary()
        assert summary["text_only"] is True
        assert summary["has_vision_tower"] is True
        assert summary["nlayer"] == 2
        assert summary["layer_type_counts"] == {
            "full_attention": 1,
            "linear_attention": 1,
        }
        assert summary["vision_weights"] == 0
        try:
            model.generate([1, 2, 3])
        except NotImplementedError as exc:
            assert "not implemented yet" in str(exc)
        else:
            raise AssertionError("Qwen3.5 skeleton should not implement generation yet")


if __name__ == "__main__":
    test_qwen2_compatible_config()
    test_nested_text_config_and_rope_parameters()
    test_reject_qwen3_5_for_qwen2_loader()
    test_real_qwen3_5_config_file()
    test_registry_supports_explicit_model_types()
    test_create_model_builds_qwen3_5_skeleton()
    print("\033[92mTest passed!\033[0m\n")
