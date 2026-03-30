#pragma once

#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"

#include <memory>
#include <vector>

namespace wginfer::models::qwen2 {

// 模型元数据
struct ModelMeta {
    wginferDataType_t dtype;
    size_t nlayer;     // 层数， 决定要跑多少层  28
    size_t hs;         // hidden size    1536 q_proj= hidden_size = num_head * head_dimension
    size_t nh;         // num heads   12
    size_t nkvh;       // num kv heads      2
    size_t dh;         // head dimension， 与nh、nkvh共同决定attention的head结构  128
    size_t di;         // intermediate size， 决定MLP中间层的大小，是FFN/MLP内部临时扩张后的维度   8960
    size_t maxseq;     // max sequence length，一次对话最多多少个prompt，决定kvcache的大小 131072，决定上下文窗口的上限
    size_t voc;        // vocabulary size， 决定embedding的长度   151936
    float epsilon;     // RMS norm epsilon
    float theta;       // RoPE theta
    int64_t end_token; // end token id    151643
};

// 模型权重
struct ModelWeights {
    // global 权重，全模型只有一份
    tensor_t in_embed;   // [voc, hs] embedding词表 model.embed_tokens.weight : [151936, 1536]
    tensor_t out_embed;  // [voc, hs] lm_head.weight : [151936, 1536]
    tensor_t out_norm_w; // [hs]   model.norm.weight : [1536]

    // 每层的权重
    // qwen2一层大致包括：attention norm；q/k/v/o四个投影；mlp norm；gate/up/down三个mlp投影
    // 层间主干张量始终是 [4, 1536]
    // attention 内部会展开成 [4, 12, 128] / [4, 2, 128]
    // 以一个token=4的prompt为例，层间主干张量始终是 [4, 1536]
    // attention 内部会展开成 [4, 12, 128] / [4, 2, 128]
    // FFN 内部会扩成 [4, 8960]FFN 内部会扩成 [4, 8960]
    std::vector<tensor_t> attn_norm_w; // [nlayer] x [hs]     input_layernorm.weight : [1536]
    std::vector<tensor_t> attn_q_w;    // [nlayer] x [nh * dh, hs]      q_proj.weight : [1536, 1536]
    std::vector<tensor_t> attn_q_b;    // [nlayer] x [nh * dh] (可能为空)      q_proj.bias : [1536]
    std::vector<tensor_t> attn_k_w;    // [nlayer] x [nkvh * dh, hs]         k_proj.weight : [256, 1536]
    std::vector<tensor_t> attn_k_b;    // [nlayer] x [nkvh * dh] (可能为空)     k_proj.bias : [256]
    std::vector<tensor_t> attn_v_w;    // [nlayer] x [nkvh * dh, hs]         v_proj.weight : [256, 1536]
    std::vector<tensor_t> attn_v_b;    // [nlayer] x [nkvh * dh] (可能为空)   v_proj.bias : [256]
    std::vector<tensor_t> attn_o_w;    // [nlayer] x [hs, nh * dh]         o_proj.weight : [1536, 1536]

    std::vector<tensor_t> mlp_norm_w; // [nlayer] x [hs]        post_attention_layernorm.weight : [1536]
    std::vector<tensor_t> mlp_gate_w; // [nlayer] x [di, hs]       gate_proj.weight : [8960, 1536]
    std::vector<tensor_t> mlp_up_w;   // [nlayer] x [di, hs]    up_proj.weight : [8960, 1536]
    std::vector<tensor_t> mlp_down_w; // [nlayer] x [hs, di]       down_proj.weight : [1536, 8960]
};

// 模型类
class Model {
private:
    ModelMeta _meta;
    ModelWeights _weights;
    wginferDeviceType_t _device_type;
    int _device_id;

    // KV Cache: 每层的 K 和 V
    std::vector<tensor_t> _k_cache; // [nlayer] x [maxseq, nkvh, dh]
    std::vector<tensor_t> _v_cache; // [nlayer] x [maxseq, nkvh, dh]
    size_t _cache_len;              // 当前 cache 长度

    // Dummy bias tensors（用于没有 bias 的层，必须全零）
    tensor_t _dummy_bias_hs;  // [hs] - 用于 o_proj, mlp_down, out_embed
    tensor_t _dummy_bias_di;  // [di] - 用于 mlp_gate, mlp_up
    tensor_t _dummy_bias_q;   // [nh * dh] - 用于 q_proj
    tensor_t _dummy_bias_kv;  // [nkvh * dh] - 用于 k_proj, v_proj
    tensor_t _dummy_bias_voc; // [voc] - 用于 out_embed

    // 临时张量（避免重复分配）
    tensor_t _x;             // 当前隐藏状态 [seqlen, hs]
    tensor_t _x_norm;        // 归一化后的隐藏状态
    tensor_t _q_flat;        // [seqlen, nh * dh]
    tensor_t _k_flat;        // [seqlen, nkvh * dh]
    tensor_t _v_flat;        // [seqlen, nkvh * dh]
    tensor_t _q;             // Query [seqlen, nh, dh]
    tensor_t _k;             // Key [seqlen, nkvh, dh]
    tensor_t _v;             // Value [seqlen, nkvh, dh]
    tensor_t _q_rope;        // [seqlen, nh, dh]
    tensor_t _k_rope_new;    // [seqlen, nkvh, dh]
    tensor_t _k_full;        // 完整的 K（包含 cache）[total_len, nkvh, dh]
    tensor_t _v_full;        // 完整的 V（包含 cache）[total_len, nkvh, dh]
    tensor_t _attn_out;      // Attention 输出 [seqlen, nh, dh]
    tensor_t _attn_proj_out; // Attention 投影输出 [seqlen, hs]
    tensor_t _x_attn;        // Attention 残差输出 [seqlen, hs]
    tensor_t _gate;          // MLP gate [seqlen, di]
    tensor_t _up;            // MLP up [seqlen, di]
    tensor_t _swiglu_out;    // SwiGLU 输出 [seqlen, di]
    tensor_t _mlp_out;       // MLP 输出 [seqlen, hs]
    tensor_t _x_mlp;         // MLP 残差输出 [seqlen, hs]
    tensor_t _logits;        // 输出 logits [seqlen, voc]
    tensor_t _pos_ids_q;     // 位置 id [seqlen]
    tensor_t _input_ids_buf; // infer 输入缓存 tokenids[ntoken]
    tensor_t _max_idx;       // argmax 索引缓存 [1]
    tensor_t _max_val;       // argmax 值缓存 [1]

    // 前向传播辅助函数
    void ensure_tensor(tensor_t &tensor, const std::vector<size_t> &shape, wginferDataType_t dtype);
    void forward_transformer_block(size_t layer_idx, tensor_t &x, size_t seqlen, size_t total_len, tensor_t pos_ids_q);
    void update_kv_cache(size_t layer_idx, tensor_t k_new, tensor_t v_new, size_t seqlen, size_t old_len);

public:
    Model(const ModelMeta &meta, wginferDeviceType_t device_type, int device_id);
    ~Model();

    ModelWeights &weights() { return _weights; }
    const ModelWeights &weights() const { return _weights; }
    const ModelMeta &meta() const { return _meta; }

    // 前向传播：返回 logits
    tensor_t forward(tensor_t input_ids, size_t seqlen, size_t total_len);

    // 推理：生成下一个 token
    int64_t infer(
        int64_t *token_ids,
        size_t ntoken,
        int top_k = 1,
        float top_p = 1.0f,
        float temperature = 1.0f);

    // 重置 KV Cache
    void reset_cache();
};

} // namespace wginfer::models::qwen2
