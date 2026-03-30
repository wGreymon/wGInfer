import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch

from test_utils import check_equal, random_tensor, zero_tensor
from wginfer.core import Ops


def torch_causal_conv1d(out, x, weight):
    seq_len, _ = x.shape
    y = torch.nn.functional.conv1d(
        x.transpose(0, 1).unsqueeze(0),
        weight,
        bias=None,
        padding=weight.shape[2] - 1,
        groups=weight.shape[0],
    )
    y = torch.nn.functional.silu(y[:, :, :seq_len])
    out.copy_(y.squeeze(0).transpose(0, 1).to(out.dtype))


def test_op_causal_conv1d(shape, kernel_size, dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu"):
    print(f"   shape {shape} kernel_size={kernel_size} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name, scale=0.2, bias=-0.1)
    weight, weight_ = random_tensor((shape[1], 1, kernel_size), dtype_name, device_name, scale=0.2, bias=-0.1)
    out, out_ = zero_tensor(shape, dtype_name, device_name)
    torch_causal_conv1d(out, x, weight)
    Ops.causal_conv1d(out_, x_, weight_)
    assert check_equal(out_, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    args = parser.parse_args()

    test_shapes = [
        ((4, 8), 4),
        ((6, 16), 3),
    ]
    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.causal_conv1d on {args.device}")
    for shape, kernel_size in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_causal_conv1d(shape, kernel_size, dtype_name, atol, rtol, args.device)

    print("\033[92mTest passed!\033[0m\n")
