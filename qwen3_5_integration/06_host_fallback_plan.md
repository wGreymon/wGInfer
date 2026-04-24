# Qwen3.5 Host Fallback 与算子替换计划

本文档记录当前 `Qwen3.5` C++ 后端中仍然存在的 host / CPU 侧辅助计算逻辑，并给出后续替换为正式 device op 的建议优先级。

参考文件：

- `src/models/qwen3_5/model.cpp`
- `src/ops/causal_conv1d/`
- `src/ops/normalize_l2/`
- `src/ops/rope_partial/`
- `src/ops/linear_attn_gates/`
- `src/ops/mul_sigmoid/`

---

## 1. 背景

当前 `Qwen3.5` 的主推理路径已经迁到 C++ backend，但在 `src/models/qwen3_5/model.cpp` 里仍然有一些 host 侧逻辑。

这些逻辑主要用于先跑通 `Qwen3.5` 的复杂结构，尤其是：

- `linear_attention`
- `full_attention`
- partial RoPE
- gated delta rule 前处理
- q/k norm
- conv1d

这些实现目前更接近“模型内部专用辅助函数”，还没有完全收敛到 `src/ops/` 下的通用算子路径。

---

## 2. 当前 host fallback 列表

### 2.1 `host_linear_conv`

位置：

- `src/models/qwen3_5/model.cpp`

作用：

```text
qkv_host
-> depthwise causal conv1d
-> silu
-> mixed_qkv_host
```

对应官方逻辑：

```text
causal_conv1d / causal_conv1d_update
```

当前问题：

- 需要把 device tensor 拷回 host
- 在 CPU 上做 conv
- 再转回 device tensor
- 会打断 GPU 推理流水线

后续替换目标：

- `ops::causal_conv1d`
- 或者针对 decode 场景实现专门的 `causal_conv1d_update` kernel

优先级：

- `P0`

原因：

这是 linear-attention 路径里的关键前处理，且当前存在明显 D2H/H2D 开销。

---

### 2.2 `normalize_l2_rows`

作用：

```text
q = normalize(q)
k = normalize(k)
```

对应官方逻辑：

```text
use_qk_l2norm_in_kernel=True
```

当前问题：

- 在 host 侧处理 q/k
- 增加 device-host 同步

后续替换目标：

- `ops::normalize_l2`
- 或者融合进 `linear_attention` kernel

优先级：

- `P0`

原因：

q/k normalize 是 linear-attention 的核心数值路径之一，既影响性能，也影响 correctness 对齐。

---

### 2.3 `g / beta` 构造

当前逻辑：

```text
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
```

位置：

- `prepareLinearInputsFromHost(...)`

当前问题：

- `A_log`、`dt_bias` 被缓存到 host
- `g / beta` 在 CPU 侧构造
- 结果再转成 runtime tensor

后续替换目标：

- `ops::linear_attn_gates`

优先级：

- `P0`

原因：

这是 linear-attention 的状态更新核心输入，频繁出现在 prefill 和 decode 路径中。

---

### 2.4 `repeat_heads`

作用：

当：

```text
linear_num_value_heads > linear_num_key_heads
```

时，把 q/k repeat 到 value heads 数量。

当前问题：

- 在 host 侧做 tensor layout 变换
- 需要额外 host 内存和拷贝

后续替换目标：

- device-side repeat / layout transform kernel
- 或融合进 `linear_attention` kernel

优先级：

- `P1`

原因：

这部分会影响性能，但相较于 conv / gates / normalize，优先级略低。

---

### 2.5 `apply_rms_norm_with_repeated_delta_weight`

作用：

用于 `full_attention` 路径中的：

```text
q_norm
k_norm
```

注意：

`Qwen3.5` 的 norm 权重是 delta 形式，实际使用时是：

```text
1.0 + weight
```

当前问题：

- 在 host 侧执行 q/k norm
- full-attention 路径需要 D2H/H2D

后续替换目标：

- 支持 delta-weight 的 RMSNorm kernel
- 或在加载权重时预先处理为 `1 + weight` 后复用现有 `ops::rms_norm`

优先级：

- `P1`

原因：

full-attention 层数量相对 linear-attention 少，但这一步对 correctness 很敏感。

---

### 2.6 `apply_partial_rope_in_place`

作用：

只对前 `rotary_dim` 维做 RoPE，其余维度保留：

```text
q[..., :rotary_dim], k[..., :rotary_dim] -> RoPE
q[..., rotary_dim:], k[..., rotary_dim:] -> passthrough
```

当前问题：

- 当前在 host 侧做 partial RoPE
- full-attention 路径需要额外拷贝

后续替换目标：

- `ops::rope_partial`

优先级：

- `P1`

原因：

已有对应 op，迁移成本相对可控，但需要严格验证和 HF 的 rotary 行为一致。

---

### 2.7 attention 输出 gate

当前逻辑：

```text
attn_output = attn_output * sigmoid(gate)
```

当前问题：

- 目前部分路径会把 attention output 拷回 host 做 gate

后续替换目标：

- `ops::mul_sigmoid`
- 或和 `o_proj` 前处理融合

优先级：

- `P1`

原因：

已有对应 op，替换后能减少一次 host 往返。

---

### 2.8 `tensorToHostF32` / `hostF32ToTensor`

作用：

作为当前 host fallback 的桥接函数：

```text
device tensor -> host float vector
host float vector -> device tensor
```

当前问题：

- 这两个函数本身不是业务算子
- 但它们暴露了当前路径仍有大量 host/device 往返

后续目标：

- 随着 host fallback 被替换，逐步减少这两个函数在主推理路径中的调用
- 最终只保留给 debug / fallback 使用

优先级：

- `P0` 作为整体优化指标

---

## 3. 替换优先级建议

### P0：优先替换

这些最影响 `Qwen3.5` 的性能和主路径稳定性：

1. `host_linear_conv` -> `ops::causal_conv1d` / `causal_conv1d_update`
2. `normalize_l2_rows` -> `ops::normalize_l2` / fused linear-attention
3. `g / beta` 构造 -> `ops::linear_attn_gates`

### P1：第二阶段替换

这些也会影响性能，但可以在 P0 稳定后推进：

1. `apply_rms_norm_with_repeated_delta_weight`
2. `apply_partial_rope_in_place`
3. attention 输出 gate -> `ops::mul_sigmoid`
4. `repeat_heads`

### P2：长期优化

1. 融合多个小算子，减少中间 tensor
2. linear-attention 专用 fused kernel
3. decode 场景专用 state update kernel
4. workspace / memory reuse 系统化

---

## 4. 替换时的注意事项

### 4.1 不能只看性能

这些 host fallback 很多是为了贴近 HF reference 行为写出来的。  
替换成 device op 时必须先保证 correctness。

建议每替换一个 op，都做：

- 单算子对齐
- 单层输出对齐
- 首 token 对齐
- 多步 token 对齐

### 4.2 优先保证 shape 语义一致

尤其是：

- `conv1d.weight` 的布局
- q/k/v 的 head layout
- `linear_num_key_heads` 与 `linear_num_value_heads` 的 repeat 关系
- partial RoPE 的 rotary_dim
- q/k norm 的 delta-weight 语义

### 4.3 decode 路径要单独验证

prefill 和 decode 的行为不完全一样，特别是：

- conv state
- recurrent state
- KV cache

所以不能只验证 prefill。

---

## 5. 当前判断

当前 `Qwen3.5` 的 C++ 后端已经完成了主链路迁移，但仍然有不少 host fallback。  
这些 fallback 让模型路径更容易先跑通，但会限制性能，也会让 CUDA / MetaX 后端难以充分发挥。

后续优化方向应该是：

```text
先保证 correctness
再逐步把 host fallback 替换成 device op
最后做 fused kernel 和 workspace 优化
```

最优先的替换目标是：

```text
linear-attention 前处理链路
```

也就是：

```text
conv1d + q/k normalize + g/beta
```
