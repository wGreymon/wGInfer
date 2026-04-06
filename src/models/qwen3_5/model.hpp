#pragma once

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/gated_rms_norm/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/linear_attention/op.hpp"
#include "../../ops/mul_sigmoid/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wginfer::models::qwen3_5 {

enum class LayerType {
    LinearAttention = 0,
    FullAttention = 1,
};

struct ModelMeta {
    wginferDataType_t dtype;
    size_t nlayer;
    size_t hs;
    size_t nh;
    size_t nkvh;
    size_t dh;
    size_t di;
    size_t maxseq;
    size_t voc;
    float epsilon;
    float theta;
    int64_t end_token;
    bool tie_word_embeddings;
    int full_attention_interval;
    size_t linear_num_key_heads;
    size_t linear_num_value_heads;
    size_t linear_key_head_dim;
    size_t linear_value_head_dim;
    size_t linear_conv_kernel_dim;
    float partial_rotary_factor;
};

struct ModelWeights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;

    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;

    std::vector<tensor_t> full_attn_q_w;
    std::vector<tensor_t> full_attn_k_w;
    std::vector<tensor_t> full_attn_v_w;
    std::vector<tensor_t> full_attn_o_w;
    std::vector<tensor_t> full_attn_q_norm_w;
    std::vector<tensor_t> full_attn_k_norm_w;

    std::vector<tensor_t> linear_attn_qkv_w;
    std::vector<tensor_t> linear_attn_z_w;
    std::vector<tensor_t> linear_attn_o_w;
    std::vector<tensor_t> linear_attn_a_w;
    std::vector<tensor_t> linear_attn_b_w;
    std::vector<tensor_t> linear_attn_norm_w;
    std::vector<tensor_t> linear_attn_dt_bias;
    std::vector<tensor_t> linear_attn_a_log;
    std::vector<tensor_t> linear_attn_conv_w;
};

struct LinearPreparedTensors {
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t z;
    tensor_t g;
    tensor_t beta;
};

class Model {
private:
    ModelMeta _meta;
    ModelWeights _weights;
    std::vector<LayerType> _layer_types;
    wginferDeviceType_t _device_type;
    int _device_id;
    size_t _cache_len;

    std::vector<tensor_t> _full_k_cache;
    std::vector<tensor_t> _full_v_cache;
    std::vector<tensor_t> _linear_recurrent_state;
    std::vector<std::vector<float>> _linear_qkv_history;

    std::vector<std::vector<float>> _full_q_norm_host;
    std::vector<std::vector<float>> _full_k_norm_host;
    std::vector<std::vector<float>> _linear_a_log_host;
    std::vector<std::vector<float>> _linear_dt_bias_host;
    std::vector<std::vector<float>> _linear_conv_host;

    tensor_t makeTensor(const std::vector<size_t> &shape, wginferDataType_t dtype) const;
    std::vector<float> tensorToHostF32(tensor_t tensor) const;
    tensor_t hostF32ToTensor(
        const std::vector<float> &values,
        const std::vector<size_t> &shape,
        wginferDataType_t dtype) const;
    tensor_t hostI64ToTensor(const std::vector<int64_t> &values, const std::vector<size_t> &shape) const;
    LinearPreparedTensors prepareLinearInputsFromHost(
        const std::vector<float> &mixed_qkv_host,
        const std::vector<float> &z_host,
        const std::vector<float> &a_host,
        const std::vector<float> &b_host,
        size_t seqlen,
        size_t layer_idx) const;

    void cacheHostLayerWeight(size_t layer_idx, const std::string &name, tensor_t tensor);
    void appendToCache(tensor_t cache, tensor_t values, size_t start_pos) const;
    tensor_t finalLogitsFromHidden(tensor_t hidden, size_t seqlen) const;

    tensor_t prefillLinearAttentionLayer(tensor_t x, size_t layer_idx);
    tensor_t decodeLinearAttentionLayer(tensor_t x, size_t layer_idx);
    tensor_t prefillFullAttentionLayer(tensor_t x, size_t layer_idx, size_t start_pos);
    tensor_t decodeFullAttentionLayer(tensor_t x, size_t layer_idx, size_t start_pos);
    tensor_t prefillLogits(const int64_t *token_ids, size_t ntoken);
    tensor_t decodeOneTokenLogits(int64_t token_id);

    int64_t greedyNextToken(tensor_t logits) const;

public:
    Model(
        const ModelMeta &meta,
        const std::vector<LayerType> &layer_types,
        wginferDeviceType_t device_type,
        int device_id);
    ~Model() = default;

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = delete;
    Model &operator=(Model &&) = delete;

    const ModelMeta &meta() const;
    const std::vector<LayerType> &layerTypes() const;
    wginferDeviceType_t deviceType() const;
    int deviceId() const;
    ModelWeights &weights();
    const ModelWeights &weights() const;

    void setWeight(const std::string &name, tensor_t tensor);
    void setLayerWeight(const std::string &name, size_t layer_idx, tensor_t tensor);

    void reset_cache();
    tensor_t forwardLogits(int64_t *token_ids, size_t ntoken);
    int64_t infer(
        int64_t *token_ids,
        size_t ntoken,
        int top_k = 1,
        float top_p = 1.0f,
        float temperature = 1.0f);
};

} // namespace wginfer::models::qwen3_5
