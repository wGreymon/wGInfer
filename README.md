## Overview

wGInfer 是一个面向大模型推理场景的多平台、多模型推理框架，目标是提供统一的部署接口、更灵活的模型接入方式，以及从离线测试到简单web在线服务的完整使用路径。当前默认支持 `CPU + NVIDIA` 双平台，`MetaX` 通过显式开关启用。

当前仓库已经具备这些能力：

- 支持多平台后端，便于在不同硬件环境下部署与验证
- 支持多模型扩展，当前已接入 [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- 已接入 `Qwen3.5-4B` 的 `text-only` 推理路径，并完成与 `Transformers` 的首 token 及多步 token 对齐验证
- 提供 Python 接口，方便集成到训练后处理、评测与业务流程中
- 提供测试、基准脚本，以及聊天服务、CLI、Web 示例

当前 `Qwen3.5` 的接入方向以 `text-only` 为主：即使模型原始仓库包含 vision tower，当前实现也只围绕 language model 部分推进。


## Roadmap

这部分不再只是“功能清单”，而是当前阶段的长期开发地图。整体原则如下：

1. 当前只长期维护两条模型路线：`Qwen2` 与 `Qwen3.5`
2. 先补 correctness 与主推理链路，再做服务化与调度优化
3. 吸收工业推理框架（如 `vLLM`）在调度、缓存、服务引擎上的优点，但不盲目追求一次性铺开所有能力
4. 当前阶段不以“继续接更多模型”或“推进多模态”作为重点

优先级说明：

- `P0`：当前阶段必须优先完成
- `P1`：在主链稳定后尽快推进
- `P2`：中长期方向，暂不抢占主线资源

状态说明：

- `✅` 已具备
- `△` 部分具备，仍需补齐
- `❌` 尚未开始或当前未落地

### 当前聚焦模型

当前阶段只计划长期维护和持续优化两条模型路径：

- `Qwen2`：以 [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 为主验证对象
- `Qwen3.5`：以 `Qwen3.5-4B` 的 `text-only` 推理路径为主

不再以“继续接更多模型”作为当前阶段重点，后续工作会主要围绕这两个模型展开。

### 框架基础

| Topic | Priority | Status | Notes |
|---|---|---:|---|
| Runtime abstraction | P0 | ✅ | CPU / NVIDIA / MetaX runtime API |
| Tensor / Storage / Allocator | P0 | ✅ | 基础能力已具备，allocator 仍较朴素 |
| Python bindings | P0 | ✅ | pybind11 接口已可用 |
| 自动化测试体系 | P0 | △ | 已有基础测试，但服务侧、长序列与回归测试仍需补强 |
| Memory reuse / workspace | P0 | △ | 模型内部已有部分复用，后续应继续系统化 |
| Benchmark / CLI / Web / Chat Server | P1 | ✅ | 已具备从离线验证到本地服务的完整使用链路 |
| Engine / Scheduler abstraction | P1 | ❌ | 后续可对齐 `vLLM` 风格的请求调度与执行引擎 |
| Request state management | P1 | ❌ | 需要为多请求并发、暂停 / 恢复、流式输出维护统一状态 |
| Continuous batching | P1 | ❌ | 服务化的重要能力，适合在主推理稳定后推进 |
| Chunked prefill | P1 | ❌ | 长上下文场景的重要优化点 |
| Async serving pipeline | P1 | ❌ | 后续服务引擎能力建设的一部分 |
| Paged KV cache infrastructure | P2 | ❌ | 后续可对齐 `vLLM` 的 paged attention / paged cache 设计 |
| Prefix cache / prompt reuse | P2 | ❌ | 适合在服务层引入，提高重复 prompt 的吞吐与时延表现 |
| Quantized inference runtime support | P2 | ❌ | 后续需要支持量化模型加载、量化权重管理与量化执行路径 |

### 算子层

| Topic | Priority | Status | Notes |
|---|---|---:|---|
| Embedding / Linear / Add / Argmax | P0 | ✅ | 基础算子已可用 |
| RMSNorm / RoPE / SwiGLU | P0 | ✅ | 已覆盖 Qwen2 主路径所需能力 |
| Self-Attention / GQA | P0 | ✅ | 已支持标准 decoder-only attention 路径 |
| Qwen3.5 相关算子 | P0 | △ | Linear-attention 路径已接入，仍有 correctness / 性能优化空间 |
| CUDA kernel 持续优化 | P1 | △ | 现有实现可继续优化吞吐、访存与融合程度 |
| Quantization format support | P1 | △ | 已接入第一版离线量化模块 |
| Quantized linear kernels | P1 | ❌ | 建议优先做 `W4A16` |
| Quantized inference operators | P2 | ❌ | 后续让量化权重直接参与计算，而不是仅在加载时反量化 |
| 更高阶 fused ops | P2 | ❌ | 如 fused norm / fused attention path，可在主链稳定后评估 |

### 模型层

#### Qwen2

| Topic | Priority | Status | Notes |
|---|---|---:|---|
| Config 解析与权重加载 | P0 | ✅ | safetensors -> internal weights |
| C++ 主推理路径 | P0 | ✅ | 已可稳定运行 |
| Transformers 对齐 | P0 | ✅ | 已可稳定对齐输出 token |
| KV cache decode | P0 | ✅ | 基础版已完成 |
| Sampling | P0 | ✅ | greedy / top-k / top-p / temperature |
| 性能优化 | P1 | △ | CPU / NVIDIA 路径可继续优化 |
| 量化推理支持 | P2 | ❌ | 当前尚未支持真正的量化 Qwen2 推理路径 |

#### Qwen3.5

| Topic | Priority | Status | Notes |
|---|---|---:|---|
| Config 解析与层类型识别 | P0 | ✅ | 已支持 `linear_attention` / `full_attention` 混合结构 |
| Text-only 推理接入 | P0 | ✅ | 已跑通主路径 |
| C++ 主推理链路 | P0 | △ | 主路径已迁到 C++ backend，仍需继续打磨 |
| Transformers 对齐 | P0 | △ | 当前主优先项，长序列 correctness 仍在优化 |
| Cache / 状态管理 | P0 | △ | full-attention cache 与 linear-attention state 已接入 |
| 性能优化 | P1 | △ | correctness 稳定后再进一步追吞吐与时延 |
| 量化推理支持 | P2 | ❌ | 当前尚未支持真正的量化 Qwen3.5 推理路径 |
| 多模态支持 | P2 | ❌ | 当前不在计划内，暂不推进 vision tower |

### 当前开发重点（建议执行顺序）

1. `Qwen3.5` 长序列 correctness 对齐
2. `Qwen3.5` C++ backend 稳定性收敛
3. 完善自动化测试与回归验证
4. `Qwen2 / Qwen3.5` 主路径性能优化
5. 引入服务引擎抽象、请求状态管理与 continuous batching
6. 从“离线量化格式支持”推进到“量化线性算子”

### 当前明确不做或暂缓

- 接入更多模型家族
- 推进 `Qwen3.5` vision tower / 多模态能力
- 引入训练相关能力
- 过早铺开所有高阶服务优化而忽略 correctness 主线

## MetaX Benchmark

下面是一组在 `MetaX` 环境上使用 `DeepSeek-R1-Distill-Qwen-1.5B` 跑出的示例基准结果，对比对象为 `Hugging Face Transformers（PyTorch backend）` 基线：

```bash
PYTHONPATH=python python test/benchmark_infer.py --model-type qwen2 --device metax --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

总体吞吐结果：

- `Transformers (PyTorch backend)`：`37.94 tok/s`
- `wGInfer`：`59.74 tok/s`
- 总体加速比：`1.57x`

分场景结果：

| Case | Transformers (PyTorch backend) mean(ms) | Transformers (PyTorch backend) tok/s | wGInfer mean(ms) | wGInfer tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| short/32 | 871.47 | 36.72 | 356.79 | 89.69 | 2.44x |
| short/64 | 1683.05 | 38.03 | 820.93 | 77.96 | 2.05x |
| short/128 | 2122.60 | 38.16 | 1115.02 | 72.64 | 1.90x |
| medium/32 | 844.13 | 37.91 | 439.44 | 72.82 | 1.92x |
| medium/64 | 1682.15 | 38.05 | 979.90 | 65.31 | 1.72x |
| medium/128 | 3360.48 | 38.09 | 2390.00 | 53.56 | 1.41x |
| long/32 | 839.00 | 38.14 | 516.82 | 61.92 | 1.62x |
| long/64 | 1680.58 | 38.08 | 1135.69 | 56.35 | 1.48x |
| long/128 | 3389.59 | 37.76 | 2707.13 | 47.28 | 1.25x |

## NVIDIA Benchmark

下面是一组在 `NVIDIA` 环境上使用 `DeepSeek-R1-Distill-Qwen-1.5B` 跑出的示例基准结果，对比对象为 `Hugging Face Transformers（PyTorch backend）` 基线：

```bash
PYTHONPATH=python python test/benchmark_infer.py --model-type qwen2 --device nvidia --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

总体吞吐结果：

- `Transformers (PyTorch backend)`：`36.94 tok/s`
- `wGInfer`：`61.42 tok/s`
- 总体加速比：`1.66x`

分场景结果：

| Case | Transformers (PyTorch backend) mean(ms) | Transformers (PyTorch backend) tok/s | wGInfer mean(ms) | wGInfer tok/s | speedup |
|---|---:|---:|---:|---:|---:|
| short/32 | 1194.02 | 26.80 | 496.63 | 64.43 | 2.40x |
| short/64 | 2409.10 | 26.57 | 1005.93 | 63.62 | 2.39x |
| short/128 | 1971.87 | 41.08 | 1278.42 | 63.36 | 1.54x |
| medium/32 | 720.10 | 44.44 | 507.07 | 63.11 | 1.42x |
| medium/64 | 1415.30 | 45.22 | 1027.08 | 62.31 | 1.38x |
| medium/128 | 2893.04 | 44.24 | 2115.26 | 60.51 | 1.37x |
| long/32 | 862.25 | 37.11 | 524.34 | 61.03 | 1.64x |
| long/64 | 1752.18 | 36.53 | 1066.19 | 60.03 | 1.64x |
| long/128 | 3702.05 | 34.58 | 2154.77 | 59.40 | 1.72x |

## Build

### Default

默认构建 `CPU + NVIDIA` 双平台。其余平台需显式开启。

```bash
cmake -S . -B build-cmake
cmake --build build-cmake -j
```

### CPU Only
若只有CPU，则关闭显卡选项
```bash
cmake -S . -B build-cmake-cpu -DWGINFER_ENABLE_NVIDIA=OFF -DWGINFER_ENABLE_METAX=OFF
cmake --build build-cmake-cpu -j
```

### MetaX

```bash
cmake -S . -B build-cmake-metax -DWGINFER_ENABLE_METAX=ON
cmake --build build-cmake-metax -j
```

## Python Binding

```bash
cd python
python setup.py build_ext --inplace
cd ..
```

## Test

### CPU

```bash
PYTHONPATH=python python test/test_runtime.py --device cpu
PYTHONPATH=python python test/test_tensor.py --device cpu
PYTHONPATH=python python test/ops/linear.py --device cpu --dtype f32
```

### NVIDIA

```bash
PYTHONPATH=python python test/ops/linear.py --device nvidia
```

### METAX

```bash
PYTHONPATH=python python test/ops/linear.py --device metax
```

### Inference verification
**不推荐使用cpu验证**，目前cpu端未针对矩阵进行向量化优化，推理速度极度缓慢

使用 `test_infer.py --test` 可以对比 `Hugging Face Transformers` 与 `wGInfer` 的输出 token 是否一致：

```bash
PYTHONPATH=python python test/test_infer.py --model-type qwen2 --device nvidia --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test
```

如果在 `MetaX` 环境中运行，则使用：

```bash
PYTHONPATH=python python test/test_infer.py --model-type qwen2 --device metax --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test
```

### Qwen3.5 text-only verification

当前 `Qwen3.5` 主要支持 `text-only` 路径。最小验证命令示例：

```bash
PYTHONPATH=python python test/test_infer.py --model-type qwen3_5 --device nvidia --model ~/model_pkg/Qwen3.5-4B --prompt "你好，请简单介绍一下你自己" --max_steps 20
```

如果需要查看首 token logits、分层状态和 linear-attention 中间量，可以增加调试开关：

```bash
PYTHONPATH=python python test/test_infer.py --model-type qwen3_5 --device nvidia --model ~/model_pkg/Qwen3.5-4B --prompt "你好，请简单介绍一下你自己" --max_steps 1 --debug-first-step --debug-layers --debug-linear-layer
```

## Web Demo

先安装服务端依赖：

```bash
python -m pip install fastapi uvicorn
```

启动本地聊天服务：

```bash
PYTHONPATH=python python app/chat_server.py --model-type qwen2 --device nvidia --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

然后在浏览器中打开：

```text
http://127.0.0.1:8000/
```

如果需要命令行客户端访问同一个服务，也可以执行：

```bash
PYTHONPATH=python python app/chat_cli.py --url http://127.0.0.1:8000/v1/chat/completions --model wginfer-qwen2
```
