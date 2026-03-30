import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch

from test_utils import check_equal, random_tensor, zero_tensor
from wginfer.core import Ops


def torch_gated_rms_norm(out, x, gate, weight, eps):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    y = normed * weight.float()
    y = y * torch.nn.functional.silu(gate.float())
    out.copy_(y.to(out.dtype))


def test_op_gated_rms_norm(shape, dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu"):
    print(f"   shape {shape} dtype <{dtype_name}>")
    x, x_ = random_tensor(shape, dtype_name, device_name, scale=0.2, bias=-0.1)
    gate, gate_ = random_tensor(shape, dtype_name, device_name, scale=0.2, bias=-0.1)
    weight, weight_ = random_tensor((shape[1],), dtype_name, device_name, scale=0.2, bias=1.0)
    out, out_ = zero_tensor(shape, dtype_name, device_name)
    eps = 1e-6
    torch_gated_rms_norm(out, x, gate, weight, eps)
    Ops.gated_rms_norm(out_, x_, gate_, weight_, eps)
    assert check_equal(out_, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    args = parser.parse_args()

    test_shapes = [(4, 8), (6, 16)]
    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.gated_rms_norm on {args.device}")
    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_gated_rms_norm(shape, dtype_name, atol, rtol, args.device)

    print("\033[92mTest passed!\033[0m\n")
