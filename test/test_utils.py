import torch
from wginfer.core import DataType
from wginfer.core import DeviceType
from wginfer.core import MemcpyKind
from wginfer.core import RuntimeAPI
from wginfer.core import Tensor


def torch_baseline_device(device_name: str, device_id=0):
    if device_name in {"nvidia", "metax"}:
        return torch_device(device_name, device_id)
    return torch.device("cpu")


def torch_to_wginfer_memcpy_kind(torch_tensor: torch.Tensor, dst_device_name: str):
    src_is_cpu = torch_tensor.device.type == "cpu"
    dst_is_cpu = dst_device_name == "cpu"
    if src_is_cpu and dst_is_cpu:
        return MemcpyKind.D2D
    if src_is_cpu and not dst_is_cpu:
        return MemcpyKind.H2D
    if (not src_is_cpu) and dst_is_cpu:
        return MemcpyKind.D2H
    return MemcpyKind.D2D


def wginfer_to_torch_memcpy_kind(src_device_type: DeviceType, torch_tensor: torch.Tensor):
    src_is_cpu = src_device_type == DeviceType.CPU
    dst_is_cpu = torch_tensor.device.type == "cpu"
    if src_is_cpu and dst_is_cpu:
        return MemcpyKind.D2D
    if src_is_cpu and not dst_is_cpu:
        return MemcpyKind.H2D
    if (not src_is_cpu) and dst_is_cpu:
        return MemcpyKind.D2H
    return MemcpyKind.D2D


def host_to_wginfer_memcpy_kind(device_name: str):
    if device_name == "cpu":
        return MemcpyKind.D2D
    return MemcpyKind.H2D


def wginfer_to_host_memcpy_kind(device_type: DeviceType):
    if device_type == DeviceType.CPU:
        return MemcpyKind.D2D
    return MemcpyKind.D2H


def random_tensor(
    shape, dtype_name, device_name, device_id=0, scale=None, bias=None
) -> tuple[torch.Tensor, Tensor]:
    torch_tensor = torch.rand(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_baseline_device(device_name, device_id),
    )
    if scale is not None:
        torch_tensor *= scale
    if bias is not None:
        torch_tensor += bias

    wginfer_tensor = Tensor(
        shape,
        dtype=wginfer_dtype(dtype_name),
        device=wginfer_device(device_name),
        device_id=device_id,
    )

    api = RuntimeAPI(wginfer_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        wginfer_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_wginfer_memcpy_kind(torch_tensor, device_name),
    )

    return torch_tensor, wginfer_tensor


def random_int_tensor(shape, device_name, dtype_name="i64", device_id=0, low=0, high=2):
    torch_tensor = torch.randint(
        low,
        high,
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_baseline_device(device_name, device_id),
    )

    wginfer_tensor = Tensor(
        shape,
        dtype=wginfer_dtype(dtype_name),
        device=wginfer_device(device_name),
        device_id=device_id,
    )

    api = RuntimeAPI(wginfer_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        wginfer_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_wginfer_memcpy_kind(torch_tensor, device_name),
    )

    return torch_tensor, wginfer_tensor


def zero_tensor(
    shape, dtype_name, device_name, device_id=0
) -> tuple[torch.Tensor, Tensor]:
    torch_tensor = torch.zeros(
        shape,
        dtype=torch_dtype(dtype_name),
        device=torch_baseline_device(device_name, device_id),
    )

    wginfer_tensor = Tensor(
        shape,
        dtype=wginfer_dtype(dtype_name),
        device=wginfer_device(device_name),
        device_id=device_id,
    )

    api = RuntimeAPI(wginfer_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        wginfer_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_wginfer_memcpy_kind(torch_tensor, device_name),
    )

    return torch_tensor, wginfer_tensor


def arrange_tensor(
    start, end, device_name, device_id=0
) -> tuple[torch.Tensor, Tensor]:
    torch_tensor = torch.arange(start, end, device=torch_baseline_device(device_name, device_id))
    wginfer_tensor = Tensor(
        (end - start,),
        dtype=wginfer_dtype("i64"),
        device=wginfer_device(device_name),
        device_id=device_id,
    )

    api = RuntimeAPI(wginfer_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        wginfer_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_wginfer_memcpy_kind(torch_tensor, device_name),
    )

    return torch_tensor, wginfer_tensor


def check_equal(
    wginfer_result: Tensor,
    torch_answer: torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
    strict=False,
):
    shape = wginfer_result.shape()
    strides = wginfer_result.strides()
    assert shape == torch_answer.shape
    assert torch_dtype(dtype_name(wginfer_result.dtype())) == torch_answer.dtype

    right = 0
    for i in range(len(shape)):
        if strides[i] > 0:
            right += strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            raise ValueError("Negative strides are not supported yet")

    tmp = torch.zeros(
        (right + 1,),
        dtype=torch_answer.dtype,
        device=torch_baseline_device(device_name(wginfer_result.device_type()), wginfer_result.device_id()),
    )
    result = torch.as_strided(tmp, shape, strides)
    api = RuntimeAPI(wginfer_result.device_type())
    api.memcpy_sync(
        result.data_ptr(),
        wginfer_result.data_ptr(),
        (right + 1) * tmp.element_size(),
        wginfer_to_torch_memcpy_kind(wginfer_result.device_type(), result),
    )

    if strict:
        if torch.equal(result, torch_answer):
            return True
    else:
        if torch.allclose(result, torch_answer, atol=atol, rtol=rtol):
            return True

    print(f"WGINFER result: \n{result}")
    print(f"Torch answer: \n{torch_answer}")
    return False


def benchmark(torch_func, wginfer_func, device_name, warmup=10, repeat=100):
    api = RuntimeAPI(wginfer_device(device_name))

    def time_op(func):
        import time

        for _ in range(warmup):
            func()
        api.device_synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        api.device_synchronize()
        end = time.time()
        return (end - start) / repeat

    torch_time = time_op(torch_func)
    wginfer_time = time_op(wginfer_func)
    print(
        f"        Torch time: {torch_time*1000:.5f} ms \n        WGINFER time: {wginfer_time*1000:.5f} ms"
    )


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    elif device_name == "metax":
        # mcPyTorch uses CUDA-compatible API; tensors are typically exposed as cuda devices.
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def wginfer_device(device_name: str):
    if device_name == "cpu":
        return DeviceType.CPU
    elif device_name == "nvidia":
        return DeviceType.NVIDIA
    elif device_name == "metax":
        return DeviceType.METAX
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def device_name(wginfer_device: DeviceType):
    if wginfer_device == DeviceType.CPU:
        return "cpu"
    elif wginfer_device == DeviceType.NVIDIA:
        return "nvidia"
    elif wginfer_device == DeviceType.METAX:
        return "metax"
    else:
        raise ValueError(f"Unsupported wginfer device: {wginfer_device}")


def torch_dtype(dtype_name: str):
    if dtype_name == "f16":
        return torch.float16
    elif dtype_name == "f32":
        return torch.float32
    elif dtype_name == "f64":
        return torch.float64
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "i32":
        return torch.int32
    elif dtype_name == "i64":
        return torch.int64
    elif dtype_name == "u32":
        return torch.uint32
    elif dtype_name == "u64":
        return torch.uint64
    elif dtype_name == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def wginfer_dtype(dtype_name: str):
    if dtype_name == "f16":
        return DataType.F16
    elif dtype_name == "f32":
        return DataType.F32
    elif dtype_name == "f64":
        return DataType.F64
    elif dtype_name == "bf16":
        return DataType.BF16
    elif dtype_name == "i32":
        return DataType.I32
    elif dtype_name == "i64":
        return DataType.I64
    elif dtype_name == "u32":
        return DataType.U32
    elif dtype_name == "u64":
        return DataType.U64
    elif dtype_name == "bool":
        return DataType.BOOL
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def dtype_name(wginfer_dtype: DataType):
    if wginfer_dtype == DataType.F16:
        return "f16"
    elif wginfer_dtype == DataType.F32:
        return "f32"
    elif wginfer_dtype == DataType.F64:
        return "f64"
    elif wginfer_dtype == DataType.BF16:
        return "bf16"
    elif wginfer_dtype == DataType.I32:
        return "i32"
    elif wginfer_dtype == DataType.I64:
        return "i64"
    elif wginfer_dtype == DataType.U32:
        return "u32"
    elif wginfer_dtype == DataType.U64:
        return "u64"
    elif wginfer_dtype == DataType.BOOL:
        return "bool"
    else:
        raise ValueError(f"Unsupported wginfer dtype: {wginfer_dtype}")
