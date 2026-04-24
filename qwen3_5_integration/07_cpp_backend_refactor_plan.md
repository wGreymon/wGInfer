# Qwen3.5 C++ 后端重构方案

本文档用于规划 `Qwen3.5` C++ 后端的架构调整。  
目标是在继续追 `Transformers` correctness 对齐之前，先把当前 `model.cpp` 中混在一起的职责拆清楚，让后续定位、验证和替换 host fallback 更可控。

当前参考文件：

- `src/models/qwen3_5/model.hpp`
- `src/models/qwen3_5/model.cpp`
- `qwen3_5/modeling_qwen3_5.py`
- `qwen3_5_integration/01_model_structure.md`
- `qwen3_5_integration/06_host_fallback_plan.md`

---

## 1. 为什么要先重构

当前 `src/models/qwen3_5/model.cpp` 已经能跑通主推理路径，但职责比较集中：

- 模型主流程调度
- full-attention 层实现
- linear-attention 层实现
- MLP 路径
- cache 管理
- host fallback
- host/device tensor 转换
- sampling

这些逻辑混在同一个文件和几个较大的函数里，会带来几个问题：

1. 不容易判断问题来自模型结构还是算子实现
2. 不容易和 Hugging Face 的模块结构逐块对齐
3. 后续替换 host fallback 时风险较高
4. 不利于后续做 CUDA / MetaX / quantized kernel 优化

因此建议先做一轮“结构性重构”。

---

## 2. 重构原则

### 2.0 当前优先级重新确认

当前阶段的主线目标已经明确为：

1. **correctness 尽量对齐 Hugging Face**
2. **代码结构更干净、更好懂、更便于维护**

因此当前不建议把优化顺序放成：

```text
先追性能 > 边改边猜 correctness
```

更合理的顺序应该是：

```text
结构清晰 > correctness 对齐 > 性能优化
```

这意味着：

- 先让 C++ 模块边界和 HF 模块边界对应起来
- 先让 full-attention / linear-attention / MLP / cache 逻辑更容易逐块对比
- 再逐步替换 host fallback
- 最后再追更高性能和更工业化的执行路径

### 2.1 第一阶段不改变行为

第一阶段目标不是优化性能，也不是一次性修 correctness，而是：

```text
保持输出行为不变，只把职责边界拆清楚
```

这能避免重构和数值修复混在一起，导致无法判断问题来源。

### 2.2 结构尽量贴近 Hugging Face 模块

HF 的 text-only 路径大致是：

```text
Qwen3_5TextModel
  -> Qwen3_5DecoderLayer
      -> Qwen3_5Attention       # full_attention
      -> Qwen3_5GatedDeltaNet   # linear_attention
      -> Qwen3_5MLP
```

C++ 后端也应该形成类似边界：

```text
Model
  -> runDecoderLayer
      -> runFullAttention
      -> runLinearAttention
      -> runMLP
```

### 2.3 模型层只负责组织流程

长期目标是：

- 模型层负责调度和状态
- 算子层负责具体计算

当前可以暂时保留 host fallback，但要把它们集中管理，方便逐步替换。

---

## 3. 当前职责拆分建议

### 3.1 `Model` 保留为总控

`Model` 继续负责：

- `ModelMeta`
- `ModelWeights`
- `layer_types`
- 全局 cache
- prefill / decode 总流程
- 逐层调度
- sampling

建议保留的公开接口：

```cpp
reset_cache()
forwardLogits(...)
infer(...)
setWeight(...)
setLayerWeight(...)
```

### 3.2 拆出 decoder layer 级别函数

新增内部函数：

```cpp
tensor_t prefillDecoderLayer(tensor_t x, size_t layer_idx, size_t start_pos);
tensor_t decodeDecoderLayer(tensor_t x, size_t layer_idx, size_t start_pos);
```

职责：

- 做一层 block 的整体调度
- 根据 `layer_types[layer_idx]` 分发到 full / linear attention
- 统一处理 attention residual
- 调用 MLP block

目标结构：

```text
prefillDecoderLayer:
    residual = x
    x_norm = input_norm(x)
    if linear:
        attn_out = prefillLinearAttention(...)
    else:
        attn_out = prefillFullAttention(...)
    x = residual + attn_out
    x = runMLP(x)
    return x
```

这样可以对齐 HF 的 `Qwen3_5DecoderLayer.forward()`。

### 3.3 拆出 full-attention 子路径

当前函数：

```cpp
prefillFullAttentionLayer(...)
decodeFullAttentionLayer(...)
```

建议拆成更小粒度：

```cpp
struct FullAttentionPrepared {
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t gate;
};

FullAttentionPrepared prepareFullAttentionInputs(...);
tensor_t runFullAttentionCore(...);
tensor_t runFullAttentionOutputProjection(...);
```

对应 HF：

```text
Qwen3_5Attention.forward()
```

拆分后更容易检查：

- q_proj 是否正确
- q_norm / k_norm 是否正确
- partial RoPE 是否正确
- KV cache update 是否正确
- gate 乘法是否正确

### 3.4 拆出 linear-attention 子路径

当前函数：

```cpp
prefillLinearAttentionLayer(...)
decodeLinearAttentionLayer(...)
prepareLinearInputsFromHost(...)
```

建议拆成：

```cpp
struct LinearProjectionTensors {
    tensor_t qkv;
    tensor_t z;
    tensor_t a;
    tensor_t b;
};

struct LinearPreparedTensors {
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t z;
    tensor_t g;
    tensor_t beta;
};

LinearProjectionTensors projectLinearAttentionInputs(...);
LinearPreparedTensors prepareLinearAttentionInputs(...);
tensor_t runLinearAttentionCore(...);
tensor_t runLinearAttentionOutputProjection(...);
```

对应 HF：

```text
Qwen3_5GatedDeltaNet.forward()
```

拆分重点：

- `in_proj_qkv / z / a / b`
- conv1d
- split q/k/v
- q/k normalize
- g/beta 构造
- linear_attention core
- recurrent state update
- gated RMSNorm
- out_proj

### 3.5 拆出 MLP block

新增：

```cpp
tensor_t runMLPBlock(tensor_t x, size_t layer_idx);
```

职责：

```text
residual = x
x_norm = post_attention_layernorm(x)
gate = gate_proj(x_norm)
up = up_proj(x_norm)
swiglu = swiglu(gate, up)
mlp_out = down_proj(swiglu)
return residual + mlp_out
```

对应 HF：

```text
Qwen3_5MLP.forward()
```

MLP 相对稳定，优先拆出来可以让 attention 路径更清楚。

### 3.6 拆出 cache helper

当前 cache 状态散落在 `Model` 内部：

- `_full_k_cache`
- `_full_v_cache`
- `_linear_recurrent_state`
- `_linear_qkv_history`

第一阶段可以不新建类，但建议集中 helper：

```cpp
void ensureFullAttentionCache(size_t layer_idx);
void appendFullAttentionCache(size_t layer_idx, tensor_t k, tensor_t v, size_t start_pos);
tensor_t getFullKCacheSlice(size_t layer_idx, size_t total_len);
tensor_t getFullVCacheSlice(size_t layer_idx, size_t total_len);

void updateLinearQkvHistory(size_t layer_idx, const std::vector<float>& qkv, size_t qkv_dim);
tensor_t getLinearRecurrentState(size_t layer_idx);
void setLinearRecurrentState(size_t layer_idx, tensor_t state);
```

第二阶段再考虑拆成独立 `Qwen3_5Cache` 类。

---

## 4. 建议的文件组织

### 第一阶段：仍放在当前两个文件里

先只重构函数，不急着拆文件：

```text
src/models/qwen3_5/model.hpp
src/models/qwen3_5/model.cpp
```

这样改动范围小，更容易验证。

### 第二阶段：再考虑拆文件

当行为稳定后，可以拆成：

```text
src/models/qwen3_5/model.hpp
src/models/qwen3_5/model.cpp
src/models/qwen3_5/attention.hpp
src/models/qwen3_5/attention.cpp
src/models/qwen3_5/cache.hpp
src/models/qwen3_5/cache.cpp
```

但不建议第一阶段就拆文件，避免 CMake 和符号组织增加额外复杂度。

---

## 5. 分阶段实施计划

### Phase 1：只做函数级拆分

目标：

- 不改变推理结果
- 不替换 host fallback
- 只把大函数拆成更清楚的小函数

建议动作：

1. 新增 `runMLPBlock`
2. 新增 `prefillDecoderLayer / decodeDecoderLayer`
3. 把现有 `prefillLogits / decodeOneTokenLogits` 改成调用 decoder layer helper
4. 保持现有 full / linear attention 逻辑不变

验证：

- `test_model_config.py`
- `Qwen3.5 --max_steps 1`
- `Qwen3.5 --max_steps 24`
- 记录完整 `--test` 分叉点是否保持不变

### Phase 2：拆 full-attention 路径

目标：

- 让 full-attention 逻辑和 HF 的 `Qwen3_5Attention.forward()` 对应起来

建议动作：

1. 拆 `prepareFullAttentionInputs`
2. 拆 `runFullAttentionCore`
3. 拆 `runFullAttentionProjection`
4. 暂时仍保留 host partial rope / q_norm / gate

验证：

- 单独对比 full-attention 层输出
- 首 token 对齐
- 分叉点回归

### Phase 3：拆 linear-attention 路径

目标：

- 让 linear-attention 逻辑和 HF 的 `Qwen3_5GatedDeltaNet.forward()` 对齐

建议动作：

1. 拆投影阶段
2. 拆 conv 阶段
3. 拆 q/k/v/z/g/beta 准备阶段
4. 拆 linear_attention core
5. 拆 recurrent state 更新

验证：

- 对比 linear-attention 第一层输出
- 对比 prefill 与 decode 的状态一致性
- 追踪长序列分叉点

### Phase 4：集中 cache helper

目标：

- 让 full-attention cache 和 linear-attention state 的管理更清晰

建议动作：

1. 集中 full cache append / slice
2. 集中 linear qkv history 更新
3. 集中 recurrent state get/set

验证：

- prefill + decode 结果不变
- 多步生成不退化

### Phase 5：替换 host fallback

目标：

- 用正式 `ops` 替换 host 辅助实现

优先顺序：

1. `host_linear_conv` -> `ops::causal_conv1d`
2. `normalize_l2_rows` -> `ops::normalize_l2`
3. `g / beta` -> `ops::linear_attn_gates`
4. `apply_partial_rope_in_place` -> `ops::rope_partial`
5. gate sigmoid -> `ops::mul_sigmoid`

验证：

- 每替换一个 op，就做一次单点回归
- 不建议多个 op 一起替换

---

## 6. 验证策略

每个阶段都应该至少跑：

```bash
PYTHONPATH=python python test/test_model_config.py
```

以及：

```bash
PYTHONPATH=python python test/test_infer.py \
  --model-type qwen3_5 \
  --device nvidia \
  --model ~/model_pkg/Qwen3.5-4B \
  --max_steps 1
```

如果可行，再跑：

```bash
PYTHONPATH=python python test/test_infer.py \
  --model-type qwen3_5 \
  --device nvidia \
  --model ~/model_pkg/Qwen3.5-4B \
  --max_steps 24 \
  --test
```

完整 `--test` 当前不一定通过，但应该记录：

- 第一个分叉 token
- reference top-k
- runtime top-k
- logits 差异

## 6.1 当前阶段明确暂缓的事项

如果当前目标是 correctness + clean code，那么以下事项不应该抢占主线优先级：

- 接入更多模型家族
- 推进 `Qwen3.5` vision tower / 多模态完整支持
- 做 continuous batching
- 做 paged KV cache / paged attention
- 做复杂 scheduler / serving engine
- 做 quantized kernel
- 过早追 benchmark 极值

这些方向可以保留在总体 roadmap 中，但不作为当前 `Qwen3.5` C++ 后端重构主线。

---

## 7. 重构完成后的目标结构

最终希望 `model.cpp` 的主流程接近：

```cpp
tensor_t Model::prefillLogits(...) {
    x = embed(input_ids);
    for layer_idx in layers:
        x = prefillDecoderLayer(x, layer_idx, 0);
    return finalLogitsFromHidden(x);
}

tensor_t Model::decodeOneTokenLogits(...) {
    x = embed(token_id);
    for layer_idx in layers:
        x = decodeDecoderLayer(x, layer_idx, cache_len);
    return finalLogitsFromHidden(x);
}
```

而单层逻辑接近：

```cpp
tensor_t Model::prefillDecoderLayer(tensor_t x, size_t layer_idx, size_t start_pos) {
    residual = x;
    x_norm = inputNorm(x, layer_idx);

    if (layer_types[layer_idx] == LinearAttention) {
        attn_out = prefillLinearAttention(x_norm, layer_idx);
    } else {
        attn_out = prefillFullAttention(x_norm, layer_idx, start_pos);
    }

    x = add(residual, attn_out);
    x = runMLPBlock(x, layer_idx);
    return x;
}
```

这样之后，对齐 HF 会更直接：

- `prefillFullAttention` 对齐 `Qwen3_5Attention.forward`
- `prefillLinearAttention` 对齐 `Qwen3_5GatedDeltaNet.forward`
- `runMLPBlock` 对齐 `Qwen3_5MLP.forward`
- `prefillDecoderLayer` 对齐 `Qwen3_5DecoderLayer.forward`

---

## 8. 当前建议

建议先执行：

```text
Phase 1：只拆函数，不改行为
```

不要一开始就替换 host fallback，也不要一开始就拆文件。  
先让代码结构变得和 HF 模块结构更接近，再逐步做 correctness 和性能优化。
