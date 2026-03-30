from .._wginfer.models import Qwen2Meta
from .._wginfer.models import Qwen2Model

try:
    from .._wginfer.models import Qwen3_5LayerType
    from .._wginfer.models import Qwen3_5Meta
    from .._wginfer.models import Qwen3_5Model
except ImportError:
    Qwen3_5LayerType = None
    Qwen3_5Meta = None
    Qwen3_5Model = None

from ._config import UnsupportedModelConfigError
from .qwen2 import Qwen2
from .qwen3_5 import Qwen3_5
from .registry import create_model
from .registry import get_model_class
from .registry import supported_model_types

__all__ = [
    "Qwen2",
    "Qwen3_5",
    "Qwen2Meta",
    "Qwen2Model",
    "UnsupportedModelConfigError",
    "create_model",
    "get_model_class",
    "supported_model_types",
]

if Qwen3_5LayerType is not None:
    __all__.extend([
        "Qwen3_5LayerType",
        "Qwen3_5Meta",
        "Qwen3_5Model",
    ])
