import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch

from test_utils import check_equal, random_tensor, zero_tensor
from wginfer.core import Ops


def torch_linear_attention(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
):
    seqlen, nhead, kdim = q.shape
    vdim = v.shape[2]
    state = (
        torch.zeros((nhead, kdim, vdim), dtype=q.dtype, device=q.device)
        if initial_state is None
        else initial_state.clone()
    )

    for i in range(seqlen):
        for h in range(nhead):
            state[h] = state[h] * torch.exp(g[i, h])
            kv_mem = torch.sum(state[h] * k[i, h][:, None], dim=0)
            delta = (v[i, h] - kv_mem) * beta[i, h]
            state[h] = state[h] + k[i, h][:, None] * delta[None, :]
            out[i, h] = torch.sum(state[h] * q[i, h][:, None], dim=0)

    if final_state is not None:
        final_state.copy_(state)


def test_op_linear_attention(
    seqlen,
    nhead,
    kdim,
    vdim,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
):
    print(
        f"   seqlen={seqlen} nhead={nhead} kdim={kdim} vdim={vdim} dtype <{dtype_name}>"
    )
    q, q_ = random_tensor((seqlen, nhead, kdim), dtype_name, device_name, scale=0.2, bias=-0.1)
    k, k_ = random_tensor((seqlen, nhead, kdim), dtype_name, device_name, scale=0.2, bias=-0.1)
    v, v_ = random_tensor((seqlen, nhead, vdim), dtype_name, device_name, scale=0.2, bias=-0.1)
    g, g_ = random_tensor((seqlen, nhead), dtype_name, device_name, scale=0.1, bias=-0.2)
    beta, beta_ = random_tensor((seqlen, nhead), dtype_name, device_name, scale=0.2, bias=0.1)

    out, out_ = zero_tensor((seqlen, nhead, vdim), dtype_name, device_name)
    final_state, final_state_ = zero_tensor((nhead, kdim, vdim), dtype_name, device_name)
    torch_linear_attention(out, q, k, v, g, beta, final_state=final_state)
    Ops.linear_attention(out_, q_, k_, v_, g_, beta_, None, final_state_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)
    assert check_equal(final_state_, final_state, atol=atol, rtol=rtol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)
    args = parser.parse_args()

    test_shapes = [
        (2, 2, 4, 3),
        (4, 3, 8, 5),
    ]
    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.linear_attention on {args.device}")
    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_linear_attention(*shape, dtype_name, atol, rtol, args.device)

    print("\033[92mTest passed!\033[0m\n")
