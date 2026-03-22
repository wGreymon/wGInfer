from .._wginfer.core import DataType
from .._wginfer.core import DeviceType
from .._wginfer.core import MemcpyKind
from .._wginfer.core import Ops
from .._wginfer.core import RuntimeAPI
from .._wginfer.core import Tensor

Stream = int

__all__ = [
    "DataType",
    "DeviceType",
    "MemcpyKind",
    "Ops",
    "RuntimeAPI",
    "Tensor",
    "Stream",
]
