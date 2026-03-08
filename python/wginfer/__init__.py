from ._wginfer import DataType
from ._wginfer import DeviceType
from ._wginfer import MemcpyKind
from ._wginfer import Ops
from ._wginfer import RuntimeAPI
from ._wginfer import Tensor
from . import models
from .models import *

Stream = int

__all__ = [
    "RuntimeAPI",
    "DeviceType",
    "DataType",
    "MemcpyKind",
    "Stream",
    "Tensor",
    "Ops",
    "models",
]
