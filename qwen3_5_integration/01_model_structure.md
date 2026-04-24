# Qwen3.5 模型结构与完整 Forward 路径

本文档基于本地拷贝的 Hugging Face 官方实现：

- `qwen3_5/modeling_qwen3_5.py`
- `qwen3_5/configuration_qwen3_5.py`

目标是把 `Qwen3.5` 的结构从源码中抽出来，形成后续接入 `wGInfer` C++ 后端时可以直接参考的结构说明。

---

## 1. 整体分层

从源码看，`Qwen3.5` 至少可以分成三层理解：

```text
Qwen3_5ForCausalLM
  -> Qwen3_5Model 或 Qwen3_5TextModel
      -> Qwen3_5DecoderLayer x N
          -> linear_attention 或 full_attention
          -> MLP
```

但要注意，源码里存在两条入口：

1. `Qwen3_5ForCausalLM`
   - 纯文本 causal LM 入口
   - 里面直接使用 `Qwen3_5TextModel`

2. `Qwen3_5Model`
   - 多模态 wrapper
   - 包含 `visual` 和 `language_model`
   - 如果有图片 / 视频输入，会先跑视觉分支，再把视觉特征写回文本 embedding

当前 `wGInfer` 的接入目标是：

```text
Qwen3.5 text-only
```

也就是说，当前只实现：

```text
input_ids
-> language_model
-> logits
```

不实现：

```text
pixel_values / video_values
-> visual
-> multimodal fusion
```

---

## 2. 关键类说明

### 2.1 `Qwen3_5DynamicCache`

这个类专门处理 `Qwen3.5` 的混合缓存。

它同时维护两类状态：

#### Full-attention 层

使用标准 KV cache：

```text
key_cache[layer_idx]
value_cache[layer_idx]
```

形状大致是：

```text
[batch_size, num_heads, seq_len, head_dim]
```

#### Linear-attention 层

不使用传统 KV cache，而是维护：

```text
conv_states[layer_idx]
recurrent_states[layer_idx]
```

源码注释里的形状语义是：

```text
conv_states:      [batch_size, d_inner, d_conv]
recurrent_states: [batch_size, d_inner, d_state]
```

这说明 `Qwen3.5` 的 cache 不是一套 KV cache 打天下，而是：

- full-attention 层用 KV cache
- linear-attention 层用 conv state + recurrent state

这也是 `Qwen3.5` 比 `Qwen2` 难实现的核心原因之一。

---

### 2.2 `Qwen3_5TextModel`

这是纯文本主干模型。

初始化结构：

```text
embed_tokens
layers: Qwen3_5DecoderLayer x num_hidden_layers
norm
rotary_emb
```

可以理解为：

```text
文本 token
-> embedding
-> N 层 mixed decoder layer
-> final RMSNorm
-> last_hidden_state
```

这个类不负责最终 vocab logits，最终 logits 在 `Qwen3_5ForCausalLM` 的 `lm_head` 里做。

---

### 2.3 `Qwen3_5DecoderLayer`

这是最重要的单层 block。

初始化时根据 `config.layer_types[layer_idx]` 决定这一层属于哪种 attention：

```text
如果 layer_type == "linear_attention":
    self.linear_attn = Qwen3_5GatedDeltaNet(...)

如果 layer_type == "full_attention":
    self.self_attn = Qwen3_5Attention(...)
```

每层公共部分：

```text
input_layernorm
post_attention_layernorm
mlp
```

所以 `DecoderLayer` 的通用结构是：

```text
输入 hidden_states
  -> input_layernorm
  -> token mixer
       linear_attention 或 full_attention
  -> residual add
  -> post_attention_layernorm
  -> MLP
  -> residual add
输出 hidden_states
```

其中 token mixer 根据层类型分叉。

---

## 3. Full-Attention 层结构

`full_attention` 路径由 `Qwen3_5Attention` 实现。

### 3.1 初始化模块

它包含：

```text
q_proj
k_proj
v_proj
o_proj
q_norm
k_norm
```

注意一个关键点：

```text
q_proj 输出维度 = num_attention_heads * head_dim * 2
```

所以 `q_proj` 输出会被拆成两部分：

```text
query_states, gate = chunk(q_proj(...), 2)
```

也就是说，full-attention 路径里不仅有 query，还有一个 gate。

### 3.2 Forward 顺序

`Qwen3_5Attention.forward()` 的核心顺序可以整理为：

```text
输入 hidden_states
  -> q_proj(hidden_states)
  -> 拆成 query_states 和 gate
  -> q_norm(query_states)
  -> k_proj(hidden_states)
  -> k_norm(key_states)
  -> v_proj(hidden_states)
  -> apply_rotary_pos_emb(query_states, key_states)
  -> 如果 past_key_values 存在，更新 KV cache
  -> attention_interface(query, key, value, attention_mask)
  -> reshape attention output
  -> attention output * sigmoid(gate)
  -> o_proj
输出 attn_output
```

用更短的结构图表示：

```text
x
-> q_proj -> query + gate
-> q_norm(query)
-> k_proj -> k_norm(key)
-> v_proj
-> partial RoPE
-> KV cache update
-> self attention
-> gate modulation
-> o_proj
```

### 3.3 和 Qwen2 的不同

相比 `Qwen2` 的普通 attention，`Qwen3.5` 的 full-attention 多了：

- `q_norm`
- `k_norm`
- `q_proj` 同时输出 query 和 gate
- attention 输出后乘 `sigmoid(gate)`
- partial rotary 支持

---

## 4. Linear-Attention 层结构

`linear_attention` 路径由 `Qwen3_5GatedDeltaNet` 实现。

这是 `Qwen3.5` 最复杂的部分。

### 4.1 初始化模块

它包含：

```text
in_proj_qkv
in_proj_z
in_proj_b
in_proj_a
conv1d
dt_bias
A_log
norm
out_proj
```

核心结构参数包括：

```text
linear_num_key_heads
linear_num_value_heads
linear_key_head_dim
linear_value_head_dim
linear_conv_kernel_dim
```

其中：

```text
key_dim   = linear_num_key_heads   * linear_key_head_dim
value_dim = linear_num_value_heads * linear_value_head_dim
conv_dim  = key_dim * 2 + value_dim
```

### 4.2 Forward 顺序

`Qwen3_5GatedDeltaNet.forward()` 的核心顺序是：

```text
输入 hidden_states
  -> apply_mask_to_padding_states
  -> in_proj_qkv(hidden_states)
  -> transpose 到 conv 输入布局
  -> in_proj_z(hidden_states)
  -> in_proj_b(hidden_states)
  -> in_proj_a(hidden_states)
  -> causal conv1d
  -> 拆分 mixed_qkv 为 query / key / value
  -> reshape query / key / value
  -> beta = sigmoid(b)
  -> g = -exp(A_log) * softplus(a + dt_bias)
  -> 如果 value_heads 多于 key_heads，则 repeat query/key
  -> gated delta attention
       prefill: chunk_gated_delta_rule
       decode: recurrent_gated_delta_rule
  -> 更新 recurrent state
  -> gated RMSNorm(core_attn_out, z)
  -> out_proj
输出 output
```

结构图：

```text
x
-> in_proj_qkv
-> conv1d
-> split q / k / v
-> in_proj_z
-> in_proj_a
-> in_proj_b
-> build g / beta
-> gated delta rule
-> recurrent state update
-> gated RMSNorm with z
-> out_proj
```

### 4.3 Prefill 与 Decode 的区别

#### Prefill

当不是单 token cached forward 时，走：

```text
chunk_gated_delta_rule(...)
```

它会处理整段序列，并在需要时输出最后 recurrent state。

#### Decode

当：

```text
cache_params is not None
and cache_params.has_previous_state
and seq_len == 1
```

会走 cached path：

```text
causal_conv1d_update(...)
recurrent_gated_delta_rule(...)
```

此时：

- conv_state 会被更新
- recurrent_state 会被读取和更新
- 每次只处理一个新 token

### 4.4 和 full-attention 的根本区别

`full_attention` 是：

```text
QK^T -> softmax -> V
```

而 `linear_attention` 是：

```text
conv + gates + recurrent state -> gated delta rule
```

所以它不是换了一个 attention kernel，而是一套不同的状态机制。

---

## 5. MLP 结构

`Qwen3_5MLP` 比较标准，是 gated MLP / SwiGLU 风格：

```text
gate_proj(x)
up_proj(x)
act(gate_proj(x)) * up_proj(x)
down_proj(...)
```

代码结构可以概括为：

```text
down_proj(act(gate_proj(x)) * up_proj(x))
```

这和 Qwen2 的 MLP 结构比较接近。

---

## 6. RMSNorm 特点

`Qwen3_5RMSNorm` 的权重初始化和普通 RMSNorm 有个细节：

```text
weight 初始为 0
实际使用时是 1.0 + weight
```

forward 里逻辑是：

```text
output = x * rsqrt(mean(x^2) + eps)
output = output * (1.0 + weight)
```

所以在自己实现时要注意：

> checkpoint 里的 norm weight 是 delta 形式，需要在使用时加 1

这一点对 correctness 对齐很重要。

---

## 7. Qwen3_5TextModel 的完整 forward

这是当前 `wGInfer text-only` 最应该参考的路径。

### 7.1 输入

`Qwen3_5TextModel.forward()` 接收：

```text
input_ids
attention_mask
position_ids
past_key_values
inputs_embeds
use_cache
```

其中：

- `input_ids` 和 `inputs_embeds` 必须二选一
- `past_key_values` 如果为空且 `use_cache=True`，会创建 `Qwen3_5DynamicCache`

### 7.2 Embedding

如果没传 `inputs_embeds`：

```text
inputs_embeds = embed_tokens(input_ids)
```

### 7.3 创建 cache

如果需要 cache：

```text
past_key_values = Qwen3_5DynamicCache(config)
```

这个 cache 同时服务：

- full-attention 的 KV cache
- linear-attention 的 conv/recurrent state

### 7.4 构造 position_ids

text-only 时，源码会生成一个 4 维 position id：

```text
[text, temporal, height, width]
```

即使当前是纯文本，也会先扩成 4 路，然后再拆出：

```text
text_position_ids = position_ids[0]
position_ids = position_ids[1:]
```

其中：

- `text_position_ids` 用于 causal mask
- `position_ids[1:]` 用于 rotary embedding

这和多模态 M-RoPE 设计有关。

### 7.5 构造 attention mask

会构造两套 mask：

```text
causal_mask
linear_attn_mask
```

然后每一层根据 `layer_types[i]` 选择：

```text
如果当前层是 linear_attention:
    layer_mask = linear_attn_mask
否则:
    layer_mask = causal_mask
```

### 7.6 计算 rotary embedding

```text
position_embeddings = rotary_emb(hidden_states, position_ids)
```

这个 embedding 会传给每一层 full-attention。

### 7.7 逐层 forward

核心循环：

```text
hidden_states = inputs_embeds

for i, decoder_layer in enumerate(layers):
    if layer_types[i] == "linear_attention":
        layer_mask = linear_attn_mask
    else:
        layer_mask = causal_mask

    hidden_states = decoder_layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=layer_mask,
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )
```

每一层内部再根据自己的 `layer_type` 走：

- `Qwen3_5GatedDeltaNet`
- 或 `Qwen3_5Attention`

### 7.8 Final Norm

所有 decoder layer 跑完后：

```text
hidden_states = norm(hidden_states)
```

### 7.9 输出

返回：

```text
last_hidden_state
past_key_values
```

注意：

`Qwen3_5TextModel` 本身还没有计算 vocab logits。

---

## 8. Qwen3_5ForCausalLM 的完整 forward

`Qwen3_5ForCausalLM` 是纯文本语言模型入口。

初始化结构：

```text
self.model = Qwen3_5TextModel(config)
self.lm_head = Linear(hidden_size, vocab_size, bias=False)
```

完整 forward：

```text
input_ids
-> Qwen3_5TextModel
-> last_hidden_state
-> lm_head
-> logits
-> optional loss
-> CausalLMOutputWithPast
```

代码核心逻辑：

```text
outputs = self.model(...)
hidden_states = outputs.last_hidden_state
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

其中：

- `logits_to_keep` 用来控制只算最后几个 token 的 logits
- 推理时通常只需要最后一个位置的 logits
- 训练时如果传了 `labels`，会额外计算 `loss`

---

## 9. 多模态 wrapper 的 forward

如果使用 `Qwen3_5Model`，它会包含：

```text
self.visual = Qwen3_5VisionModel(...)
self.language_model = Qwen3_5TextModel(...)
```

完整多模态 forward 大致是：

```text
input_ids / inputs_embeds
如果有 image:
    pixel_values -> visual -> image_embeds
    image_embeds 写入 image placeholder 位置
如果有 video:
    pixel_values_videos -> visual -> video_embeds
    video_embeds 写入 video placeholder 位置
计算 3D position ids
language_model(inputs_embeds=融合后的 embeddings)
返回 hidden states / cache / rope_deltas
```

当前 `wGInfer` 的 `text-only` 路径不走这部分 vision 逻辑。

也就是说，当前只实现：

```text
Qwen3_5ForCausalLM / Qwen3_5TextModel 这条语言模型路径
```

而不是完整：

```text
Qwen3_5Model + visual + language_model
```

---

## 10. Text-only forward 总结

当前 `wGInfer` 应该重点对齐的 text-only forward 可以整理成：

```text
input_ids
  -> embed_tokens
  -> build DynamicCache if use_cache
  -> build position_ids
  -> build causal_mask / linear_attn_mask
  -> rotary_emb
  -> for each decoder layer:
       -> input_layernorm
       -> if layer_type == linear_attention:
              in_proj_qkv / in_proj_z / in_proj_a / in_proj_b
              conv1d
              split q/k/v
              build g / beta
              gated delta rule
              update recurrent state
              gated RMSNorm
              out_proj
          else:
              q_proj -> query + gate
              q_norm / k_norm
              v_proj
              partial RoPE
              update KV cache
              self attention
              sigmoid gate
              o_proj
       -> residual
       -> post_attention_layernorm
       -> MLP
       -> residual
  -> final norm
  -> lm_head
  -> logits
```

---

## 11. 对 wGInfer 实现的启发

### 11.1 不能把 Qwen3.5 当成普通 Qwen2

Qwen2 的一层基本是固定结构。  
Qwen3.5 必须按 `layer_types` 分流。

### 11.2 Cache 必须按层类型区分

`full_attention`：

```text
KV cache
```

`linear_attention`：

```text
conv state + recurrent state
```

### 11.3 Norm weight 要注意 delta 语义

`Qwen3_5RMSNorm` 使用：

```text
1.0 + weight
```

这和普通 `RMSNorm(weight)` 不完全一样。

### 11.4 full-attention 不是简单 Qwen2 attention

它额外有：

- `q_norm`
- `k_norm`
- `gate`
- partial RoPE

### 11.5 linear-attention 是最复杂的部分

真正难点在：

- conv state
- recurrent state
- gated delta rule
- q/k repeat
- g / beta 构造

---

## 12. 当前后续接入重点

后续继续优化 `wGInfer` 的 `Qwen3.5`，建议优先围绕：

1. 对齐 `Qwen3_5TextModel.forward`
2. 对齐 `Qwen3_5DecoderLayer.forward`
3. 对齐 `Qwen3_5GatedDeltaNet.forward`
4. 对齐 `Qwen3_5Attention.forward`
5. 再做 C++ backend 性能优化

优先级应该是：

```text
correctness > cache 稳定性 > 性能优化
```
