import argparse

import wginfer
import torch
from test_utils import *


def test_tensor(device_name: str = "cpu"):
    torch_tensor_host = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    torch_tensor = torch_tensor_host.to(torch_baseline_device(device_name))
    wginfer_tensor = wginfer.Tensor(
        (3, 4, 5), dtype=wginfer_dtype("i64"), device=wginfer_device(device_name)
    )

    # Test load
    print("===Test load===")
    wginfer_tensor.load(torch_tensor_host.data_ptr())
    wginfer_tensor.debug()
    assert wginfer_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(wginfer_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    wginfer_tensor_view = wginfer_tensor.view(6, 10)
    wginfer_tensor_view.debug()
    assert wginfer_tensor_view.shape() == torch_tensor_view.shape
    assert wginfer_tensor_view.strides() == torch_tensor_view.stride()
    assert wginfer_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(wginfer_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    wginfer_tensor_perm = wginfer_tensor.permute(2, 0, 1)
    wginfer_tensor_perm.debug()
    assert wginfer_tensor_perm.shape() == torch_tensor_perm.shape
    assert wginfer_tensor_perm.strides() == torch_tensor_perm.stride()
    assert wginfer_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(wginfer_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    wginfer_tensor_slice = wginfer_tensor.slice(2, 1, 4)
    wginfer_tensor_slice.debug()
    assert wginfer_tensor_slice.shape() == torch_tensor_slice.shape
    assert wginfer_tensor_slice.strides() == torch_tensor_slice.stride()
    assert wginfer_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(wginfer_tensor_slice, torch_tensor_slice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    args = parser.parse_args()

    test_tensor(args.device)

    print("\n\033[92mTest passed!\033[0m\n")
