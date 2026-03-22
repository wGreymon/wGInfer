## Overview

wGInfer 是一个面向大模型推理场景的多平台、多模型推理框架，目标是提供统一的部署接口、更灵活的模型接入方式，以及从离线测试到简单web在线服务的完整使用路径。当前默认支持 `CPU + NVIDIA` 双平台，`MetaX` 通过显式开关启用。

当前仓库已经具备这些能力：

- 支持多平台后端，便于在不同硬件环境下部署与验证
- 支持多模型扩展，当前已接入 [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- 提供 Python 接口，方便集成到训练后处理、评测与业务流程中
- 提供测试、基准脚本，以及聊天服务、CLI、Web 示例

## MetaX Benchmark

下面是一组在 `MetaX` 环境上使用 `DeepSeek-R1-Distill-Qwen-1.5B` 跑出的示例基准结果，对比对象为 `Hugging Face Transformers（PyTorch backend）` 基线：

```bash
PYTHONPATH=python python test/benchmark_infer.py --device metax --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
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
PYTHONPATH=python python test/benchmark_infer.py --device nvidia --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
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
PYTHONPATH=python python test/test_infer.py --device nvidia --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test
```

如果在 `MetaX` 环境中运行，则使用：

```bash
PYTHONPATH=python python test/test_infer.py --device metax --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/ --test
```

## Web Demo

先安装服务端依赖：

```bash
python -m pip install fastapi uvicorn
```

启动本地聊天服务：

```bash
PYTHONPATH=python python test/chat_server.py --device nvidia --model ~/model_pkg/DeepSeek-R1-Distill-Qwen-1.5B/
```

然后在浏览器中打开：

```text
http://127.0.0.1:8000/
```

如果需要命令行客户端访问同一个服务，也可以执行：

```bash
PYTHONPATH=python python test/chat_cli.py --url http://127.0.0.1:8000/v1/chat/completions --model wginfer-qwen2
```
