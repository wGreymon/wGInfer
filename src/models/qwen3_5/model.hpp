#pragma once

#include "wginfer.h"

#include <cstddef>
#include <cstdint>
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
};

class Model {
private:
    ModelMeta _meta;
    std::vector<LayerType> _layer_types;
    wginferDeviceType_t _device_type;
    int _device_id;

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

    void reset_cache();
};

} // namespace wginfer::models::qwen3_5
