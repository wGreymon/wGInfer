#include "model.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace wginfer::models::qwen3_5 {

namespace {

bool is_supported_layer_type(LayerType layer_type) {
    return layer_type == LayerType::LinearAttention || layer_type == LayerType::FullAttention;
}

int64_t argmax_host(const std::vector<float> &vals) {
    ASSERT(!vals.empty(), "argmax_host: input must not be empty");
    size_t best = 0;
    for (size_t i = 1; i < vals.size(); ++i) {
        if (vals[i] > vals[best]) {
            best = i;
        }
    }
    return static_cast<int64_t>(best);
}

std::vector<float> sample_logits_to_probs(
    const std::vector<float> &logits,
    std::vector<int> &indices,
    int top_k,
    float top_p,
    float temperature) {
    const size_t vocab = logits.size();
    if (top_k <= 0 || top_k > static_cast<int>(vocab)) {
        top_k = static_cast<int>(vocab);
    }
    if (top_p <= 0.0f || top_p > 1.0f) {
        top_p = 1.0f;
    }

    indices.resize(vocab);
    std::iota(indices.begin(), indices.end(), 0);
    auto by_logit_desc = [&logits](int a, int b) { return logits[a] > logits[b]; };
    if (top_k < static_cast<int>(vocab)) {
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(), by_logit_desc);
        indices.resize(top_k);
    } else {
        std::sort(indices.begin(), indices.end(), by_logit_desc);
    }

    const float inv_temp = 1.0f / temperature;
    float max_scaled = -std::numeric_limits<float>::infinity();
    for (int idx : indices) {
        max_scaled = std::max(max_scaled, logits[idx] * inv_temp);
    }

    std::vector<float> probs(indices.size(), 0.0f);
    float total = 0.0f;
    for (size_t i = 0; i < indices.size(); ++i) {
        float p = std::exp(logits[indices[i]] * inv_temp - max_scaled);
        if (!std::isfinite(p) || p < 0.0f) {
            p = 0.0f;
        }
        probs[i] = p;
        total += p;
    }
    if (total <= 0.0f) {
        probs.assign(indices.size(), 0.0f);
        probs.front() = 1.0f;
        return probs;
    }

    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t keep = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i] / total;
            keep = i + 1;
            if (cumulative >= top_p) {
                break;
            }
        }
        keep = std::max<size_t>(keep, 1);
        indices.resize(keep);
        probs.resize(keep);
    }

    return probs;
}

int64_t sample_from_logits(
    const std::vector<float> &logits,
    int top_k,
    float top_p,
    float temperature) {
    ASSERT(!logits.empty(), "sample_from_logits: logits must not be empty");
    if (temperature <= 0.0f) {
        return argmax_host(logits);
    }

    if (top_k == 1 && top_p >= 1.0f && std::abs(temperature - 1.0f) < 1e-6f) {
        return argmax_host(logits);
    }

    std::vector<int> indices;
    std::vector<float> probs = sample_logits_to_probs(logits, indices, top_k, top_p, temperature);

    thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    const int chosen = dist(rng);
    return static_cast<int64_t>(indices[static_cast<size_t>(chosen)]);
}

std::vector<float> host_linear_conv(
    const std::vector<float> &in,
    size_t seqlen,
    size_t channels,
    size_t kernel_size,
    const std::vector<float> &weight) {
    std::vector<float> out(seqlen * channels, 0.0f);
    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (size_t k = 0; k < kernel_size; ++k) {
                const size_t input_seq = seq + k;
                if (input_seq >= kernel_size - 1 && (input_seq - (kernel_size - 1)) < seqlen) {
                    const size_t src_seq = input_seq - (kernel_size - 1);
                    acc += in[src_seq * channels + c] * weight[c * kernel_size + k];
                }
            }
            out[seq * channels + c] = acc / (1.0f + std::exp(-acc));
        }
    }
    return out;
}

void normalize_l2_rows(std::vector<float> &data, size_t rows, size_t cols, float eps, float scale) {
    for (size_t row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float v = data[row * cols + col];
            sum_sq += v * v;
        }
        const float norm = std::sqrt(sum_sq);
        const float inv_norm = scale / std::max(norm, eps);
        for (size_t col = 0; col < cols; ++col) {
            data[row * cols + col] *= inv_norm;
        }
    }
}

std::vector<float> apply_rms_norm_with_repeated_delta_weight(
    const std::vector<float> &in,
    size_t rows,
    size_t cols,
    const std::vector<float> &weight_delta,
    float eps) {
    CHECK_ARGUMENT(weight_delta.size() == cols, "apply_rms_norm_with_repeated_delta_weight: weight size mismatch");
    std::vector<float> out(in.size(), 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        float variance = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float v = in[row * cols + col];
            variance += v * v;
        }
        variance /= static_cast<float>(cols);
        const float inv_std = 1.0f / std::sqrt(variance + eps);
        for (size_t col = 0; col < cols; ++col) {
            out[row * cols + col] =
                in[row * cols + col] * inv_std * (weight_delta[col] + 1.0f);
        }
    }
    return out;
}

void apply_partial_rope_in_place(
    std::vector<float> &data,
    size_t seqlen,
    size_t nhead,
    size_t head_dim,
    const std::vector<int64_t> &pos_ids,
    float theta,
    size_t rotary_dim) {
    if (rotary_dim == 0) {
        return;
    }
    const size_t half = rotary_dim / 2;
    for (size_t seq = 0; seq < seqlen; ++seq) {
        const float pos = static_cast<float>(pos_ids[seq]);
        for (size_t head = 0; head < nhead; ++head) {
            const size_t base = (seq * nhead + head) * head_dim;
            for (size_t j = 0; j < half; ++j) {
                const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(rotary_dim);
                const float phi = pos / std::pow(theta, exponent);
                const float sinv = std::sin(phi);
                const float cosv = std::cos(phi);
                const float a = data[base + j];
                const float b = data[base + j + half];
                data[base + j] = a * cosv - b * sinv;
                data[base + j + half] = b * cosv + a * sinv;
            }
        }
    }
}

std::vector<float> repeat_heads(
    const std::vector<float> &in,
    size_t seqlen,
    size_t src_heads,
    size_t dst_heads,
    size_t head_dim) {
    CHECK_ARGUMENT(dst_heads % src_heads == 0, "repeat_heads: dst_heads must be divisible by src_heads");
    const size_t repeat_factor = dst_heads / src_heads;
    std::vector<float> out(seqlen * dst_heads * head_dim, 0.0f);
    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t head = 0; head < src_heads; ++head) {
            const float *src = &in[(seq * src_heads + head) * head_dim];
            for (size_t rep = 0; rep < repeat_factor; ++rep) {
                float *dst = &out[(seq * dst_heads + head * repeat_factor + rep) * head_dim];
                std::copy(src, src + head_dim, dst);
            }
        }
    }
    return out;
}

} // namespace

Model::Model(
    const ModelMeta &meta,
    const std::vector<LayerType> &layer_types,
    wginferDeviceType_t device_type,
    int device_id)
    : _meta(meta),
      _layer_types(layer_types),
      _device_type(device_type),
      _device_id(device_id),
      _cache_len(0),
      _full_k_cache(meta.nlayer),
      _full_v_cache(meta.nlayer),
      _linear_recurrent_state(meta.nlayer),
      _linear_qkv_history(meta.nlayer),
      _full_q_norm_host(meta.nlayer),
      _full_k_norm_host(meta.nlayer),
      _linear_a_log_host(meta.nlayer),
      _linear_dt_bias_host(meta.nlayer),
      _linear_conv_host(meta.nlayer) {
    CHECK_ARGUMENT(_meta.nlayer == _layer_types.size(), "Qwen3.5 layer_types size must match nlayer");
    for (LayerType layer_type : _layer_types) {
        CHECK_ARGUMENT(is_supported_layer_type(layer_type), "Qwen3.5 layer_types contains an unsupported value");
    }

    _weights.attn_norm_w.resize(_meta.nlayer);
    _weights.mlp_norm_w.resize(_meta.nlayer);
    _weights.mlp_gate_w.resize(_meta.nlayer);
    _weights.mlp_up_w.resize(_meta.nlayer);
    _weights.mlp_down_w.resize(_meta.nlayer);

    _weights.full_attn_q_w.resize(_meta.nlayer);
    _weights.full_attn_k_w.resize(_meta.nlayer);
    _weights.full_attn_v_w.resize(_meta.nlayer);
    _weights.full_attn_o_w.resize(_meta.nlayer);
    _weights.full_attn_q_norm_w.resize(_meta.nlayer);
    _weights.full_attn_k_norm_w.resize(_meta.nlayer);

    _weights.linear_attn_qkv_w.resize(_meta.nlayer);
    _weights.linear_attn_z_w.resize(_meta.nlayer);
    _weights.linear_attn_o_w.resize(_meta.nlayer);
    _weights.linear_attn_a_w.resize(_meta.nlayer);
    _weights.linear_attn_b_w.resize(_meta.nlayer);
    _weights.linear_attn_norm_w.resize(_meta.nlayer);
    _weights.linear_attn_dt_bias.resize(_meta.nlayer);
    _weights.linear_attn_a_log.resize(_meta.nlayer);
    _weights.linear_attn_conv_w.resize(_meta.nlayer);
}

tensor_t Model::makeTensor(const std::vector<size_t> &shape, wginferDataType_t dtype) const {
    return Tensor::create(shape, dtype, _device_type, _device_id);
}

std::vector<float> Model::tensorToHostF32(tensor_t tensor) const {
    ASSERT(tensor != nullptr, "tensorToHostF32: tensor must not be null");
    wginfer::core::context().setDevice(_device_type, _device_id);
    const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();

    const size_t n = tensor->numel();
    std::vector<float> out(n, 0.0f);
    switch (tensor->dtype()) {
    case WGINFER_DTYPE_F32:
        api->memcpy_sync(
            out.data(),
            tensor->data(),
            n * sizeof(float),
            (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2H);
        return out;
    case WGINFER_DTYPE_F16: {
        std::vector<wginfer::fp16_t> tmp(n);
        api->memcpy_sync(
            tmp.data(),
            tensor->data(),
            n * sizeof(wginfer::fp16_t),
            (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2H);
        for (size_t i = 0; i < n; ++i) {
            out[i] = wginfer::utils::cast<float>(tmp[i]);
        }
        return out;
    }
    case WGINFER_DTYPE_BF16: {
        std::vector<wginfer::bf16_t> tmp(n);
        api->memcpy_sync(
            tmp.data(),
            tensor->data(),
            n * sizeof(wginfer::bf16_t),
            (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2H);
        for (size_t i = 0; i < n; ++i) {
            out[i] = wginfer::utils::cast<float>(tmp[i]);
        }
        return out;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(tensor->dtype());
    }
}

tensor_t Model::hostF32ToTensor(
    const std::vector<float> &values,
    const std::vector<size_t> &shape,
    wginferDataType_t dtype) const {
    tensor_t tensor = makeTensor(shape, dtype);
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        tensor->load(values.data());
        return tensor;
    case WGINFER_DTYPE_F16: {
        std::vector<wginfer::fp16_t> tmp(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            tmp[i] = wginfer::utils::cast<wginfer::fp16_t>(values[i]);
        }
        tensor->load(tmp.data());
        return tensor;
    }
    case WGINFER_DTYPE_BF16: {
        std::vector<wginfer::bf16_t> tmp(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            tmp[i] = wginfer::utils::cast<wginfer::bf16_t>(values[i]);
        }
        tensor->load(tmp.data());
        return tensor;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

tensor_t Model::hostI64ToTensor(const std::vector<int64_t> &values, const std::vector<size_t> &shape) const {
    tensor_t tensor = makeTensor(shape, WGINFER_DTYPE_I64);
    tensor->load(values.data());
    return tensor;
}

tensor_t Model::runAttentionInputNorm(tensor_t x, size_t layer_idx) const {
    const size_t seqlen = x->shape()[0];
    tensor_t x_norm = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::rms_norm(x_norm, x, _weights.attn_norm_w[layer_idx], _meta.epsilon);
    return x_norm;
}

LinearProjectedTensors Model::projectLinearAttentionInputs(tensor_t x_norm, size_t seqlen, size_t layer_idx) const {
    const size_t key_dim = _meta.linear_num_key_heads * _meta.linear_key_head_dim;
    const size_t value_dim = _meta.linear_num_value_heads * _meta.linear_value_head_dim;
    const size_t qkv_dim = key_dim * 2 + value_dim;

    tensor_t qkv = makeTensor({seqlen, qkv_dim}, _meta.dtype);
    tensor_t z = makeTensor({seqlen, value_dim}, _meta.dtype);
    tensor_t a = makeTensor({seqlen, _meta.linear_num_value_heads}, _meta.dtype);
    tensor_t b = makeTensor({seqlen, _meta.linear_num_value_heads}, _meta.dtype);
    ops::linear(qkv, x_norm, _weights.linear_attn_qkv_w[layer_idx], nullptr);
    ops::linear(z, x_norm, _weights.linear_attn_z_w[layer_idx], nullptr);
    ops::linear(a, x_norm, _weights.linear_attn_a_w[layer_idx], nullptr);
    ops::linear(b, x_norm, _weights.linear_attn_b_w[layer_idx], nullptr);
    return {
        qkv,
        z,
        a,
        b,
    };
}

FullProjectedTensors Model::projectFullAttentionInputs(tensor_t x_norm, size_t seqlen, size_t layer_idx) const {
    tensor_t q_proj = makeTensor({seqlen, _meta.nh * _meta.dh * 2}, _meta.dtype);
    tensor_t k_proj = makeTensor({seqlen, _meta.nkvh * _meta.dh}, _meta.dtype);
    tensor_t v_proj = makeTensor({seqlen, _meta.nkvh * _meta.dh}, _meta.dtype);
    ops::linear(q_proj, x_norm, _weights.full_attn_q_w[layer_idx], nullptr);
    ops::linear(k_proj, x_norm, _weights.full_attn_k_w[layer_idx], nullptr);
    ops::linear(v_proj, x_norm, _weights.full_attn_v_w[layer_idx], nullptr);
    return {
        q_proj,
        k_proj,
        v_proj,
    };
}

FullAttentionPreparedTensors Model::prepareFullAttentionInputsFromHost(
    const std::vector<float> &q_proj_host,
    const std::vector<float> &k_proj_host,
    const std::vector<float> &v_proj_host,
    size_t seqlen,
    size_t layer_idx,
    size_t start_pos) const {
    std::vector<float> q_host(seqlen * _meta.nh * _meta.dh, 0.0f);
    std::vector<float> gate_flat(seqlen * _meta.nh * _meta.dh, 0.0f);
    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t head = 0; head < _meta.nh; ++head) {
            const size_t src_base = (seq * _meta.nh + head) * (_meta.dh * 2);
            const size_t dst_base = (seq * _meta.nh + head) * _meta.dh;
            std::copy(
                q_proj_host.begin() + static_cast<ptrdiff_t>(src_base),
                q_proj_host.begin() + static_cast<ptrdiff_t>(src_base + _meta.dh),
                q_host.begin() + static_cast<ptrdiff_t>(dst_base));
            std::copy(
                q_proj_host.begin() + static_cast<ptrdiff_t>(src_base + _meta.dh),
                q_proj_host.begin() + static_cast<ptrdiff_t>(src_base + 2 * _meta.dh),
                gate_flat.begin() + static_cast<ptrdiff_t>(dst_base));
        }
    }

    std::vector<float> k_host = k_proj_host;
    std::vector<float> v_host = v_proj_host;

    q_host = apply_rms_norm_with_repeated_delta_weight(
        q_host,
        seqlen * _meta.nh,
        _meta.dh,
        _full_q_norm_host[layer_idx],
        _meta.epsilon);
    k_host = apply_rms_norm_with_repeated_delta_weight(
        k_host,
        seqlen * _meta.nkvh,
        _meta.dh,
        _full_k_norm_host[layer_idx],
        _meta.epsilon);

    size_t rotary_dim = static_cast<size_t>(static_cast<float>(_meta.dh) * _meta.partial_rotary_factor);
    rotary_dim -= rotary_dim % 2;
    std::vector<int64_t> pos_ids(seqlen);
    for (size_t i = 0; i < seqlen; ++i) {
        pos_ids[i] = static_cast<int64_t>(start_pos + i);
    }
    apply_partial_rope_in_place(q_host, seqlen, _meta.nh, _meta.dh, pos_ids, _meta.theta, rotary_dim);
    apply_partial_rope_in_place(k_host, seqlen, _meta.nkvh, _meta.dh, pos_ids, _meta.theta, rotary_dim);

    return {
        hostF32ToTensor(q_host, {seqlen, _meta.nh, _meta.dh}, _meta.dtype),
        hostF32ToTensor(k_host, {seqlen, _meta.nkvh, _meta.dh}, _meta.dtype),
        hostF32ToTensor(v_host, {seqlen, _meta.nkvh, _meta.dh}, _meta.dtype),
        std::move(gate_flat),
    };
}

void Model::cacheHostLayerWeight(size_t layer_idx, const std::string &name, tensor_t tensor) {
    if (name == "full_attn_q_norm_w") {
        _full_q_norm_host.at(layer_idx) = tensorToHostF32(std::move(tensor));
        return;
    }
    if (name == "full_attn_k_norm_w") {
        _full_k_norm_host.at(layer_idx) = tensorToHostF32(std::move(tensor));
        return;
    }
    if (name == "linear_attn_a_log") {
        _linear_a_log_host.at(layer_idx) = tensorToHostF32(std::move(tensor));
        return;
    }
    if (name == "linear_attn_dt_bias") {
        _linear_dt_bias_host.at(layer_idx) = tensorToHostF32(std::move(tensor));
        return;
    }
    if (name == "linear_attn_conv_w") {
        _linear_conv_host.at(layer_idx) = tensorToHostF32(std::move(tensor));
        return;
    }
}

const ModelMeta &Model::meta() const {
    return _meta;
}

const std::vector<LayerType> &Model::layerTypes() const {
    return _layer_types;
}

wginferDeviceType_t Model::deviceType() const {
    return _device_type;
}

int Model::deviceId() const {
    return _device_id;
}

ModelWeights &Model::weights() {
    return _weights;
}

const ModelWeights &Model::weights() const {
    return _weights;
}

void Model::setWeight(const std::string &name, tensor_t tensor) {
    if (name == "in_embed") {
        _weights.in_embed = tensor;
        if (_meta.tie_word_embeddings && !_weights.out_embed) {
            _weights.out_embed = tensor;
        }
        return;
    }
    if (name == "out_embed") {
        _weights.out_embed = tensor;
        return;
    }
    if (name == "out_norm_w") {
        _weights.out_norm_w = tensor;
        return;
    }
    throw std::invalid_argument("Unknown global Qwen3.5 weight: " + name);
}

void Model::setLayerWeight(const std::string &name, size_t layer_idx, tensor_t tensor) {
    CHECK_ARGUMENT(layer_idx < _meta.nlayer, "Qwen3.5 layer_idx out of range");
    if (name == "attn_norm_w") {
        _weights.attn_norm_w.at(layer_idx) = tensor;
    } else if (name == "mlp_norm_w") {
        _weights.mlp_norm_w.at(layer_idx) = tensor;
    } else if (name == "mlp_gate_w") {
        _weights.mlp_gate_w.at(layer_idx) = tensor;
    } else if (name == "mlp_up_w") {
        _weights.mlp_up_w.at(layer_idx) = tensor;
    } else if (name == "mlp_down_w") {
        _weights.mlp_down_w.at(layer_idx) = tensor;
    } else if (name == "full_attn_q_w") {
        _weights.full_attn_q_w.at(layer_idx) = tensor;
    } else if (name == "full_attn_k_w") {
        _weights.full_attn_k_w.at(layer_idx) = tensor;
    } else if (name == "full_attn_v_w") {
        _weights.full_attn_v_w.at(layer_idx) = tensor;
    } else if (name == "full_attn_o_w") {
        _weights.full_attn_o_w.at(layer_idx) = tensor;
    } else if (name == "full_attn_q_norm_w") {
        _weights.full_attn_q_norm_w.at(layer_idx) = tensor;
    } else if (name == "full_attn_k_norm_w") {
        _weights.full_attn_k_norm_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_qkv_w") {
        _weights.linear_attn_qkv_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_z_w") {
        _weights.linear_attn_z_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_o_w") {
        _weights.linear_attn_o_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_a_w") {
        _weights.linear_attn_a_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_b_w") {
        _weights.linear_attn_b_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_norm_w") {
        _weights.linear_attn_norm_w.at(layer_idx) = tensor;
    } else if (name == "linear_attn_dt_bias") {
        _weights.linear_attn_dt_bias.at(layer_idx) = tensor;
    } else if (name == "linear_attn_a_log") {
        _weights.linear_attn_a_log.at(layer_idx) = tensor;
    } else if (name == "linear_attn_conv_w") {
        _weights.linear_attn_conv_w.at(layer_idx) = tensor;
    } else {
        throw std::invalid_argument("Unknown per-layer Qwen3.5 weight: " + name);
    }

    cacheHostLayerWeight(layer_idx, name, tensor);
}

void Model::reset_cache() {
    _cache_len = 0;
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _linear_qkv_history[i].clear();
        _linear_recurrent_state[i] = nullptr;
    }
}

tensor_t Model::forwardLogits(int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen3.5 forwardLogits: token_ids must not be null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen3.5 forwardLogits: ntoken must be positive");
    reset_cache();
    tensor_t logits = prefillLogits(token_ids, ntoken);
    _cache_len = ntoken;
    return logits;
}

void Model::appendToCache(tensor_t cache, tensor_t values, size_t start_pos) const {
    ASSERT(cache != nullptr && values != nullptr, "appendToCache: tensors must not be null");
    CHECK_ARGUMENT(values->shape()[0] + start_pos <= cache->shape()[0], "appendToCache: cache overflow");

    size_t row_elems = 1;
    for (size_t dim = 1; dim < values->shape().size(); ++dim) {
        row_elems *= values->shape()[dim];
    }
    const size_t row_bytes = row_elems * values->elementSize();
    const size_t total_bytes = values->numel() * values->elementSize();

    wginfer::core::context().setDevice(_device_type, _device_id);
    const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();
    api->memcpy_sync(
        cache->data() + start_pos * row_bytes,
        values->data(),
        total_bytes,
        (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2D);
}

LinearPreparedTensors Model::prepareLinearInputsFromHost(
    const std::vector<float> &mixed_qkv_host,
    const std::vector<float> &z_host,
    const std::vector<float> &a_host,
    const std::vector<float> &b_host,
    size_t seqlen,
    size_t layer_idx) const {
    const ModelMeta &meta = _meta;
    const size_t key_dim = meta.linear_num_key_heads * meta.linear_key_head_dim;
    const size_t value_dim = meta.linear_num_value_heads * meta.linear_value_head_dim;
    const size_t qkv_dim = key_dim * 2 + value_dim;

    std::vector<float> q(seqlen * meta.linear_num_key_heads * meta.linear_key_head_dim, 0.0f);
    std::vector<float> k(seqlen * meta.linear_num_key_heads * meta.linear_key_head_dim, 0.0f);
    std::vector<float> v(seqlen * meta.linear_num_value_heads * meta.linear_value_head_dim, 0.0f);
    std::vector<float> z = z_host;

    for (size_t seq = 0; seq < seqlen; ++seq) {
        const float *src = &mixed_qkv_host[seq * qkv_dim];
        std::copy(src, src + key_dim, &q[seq * key_dim]);
        std::copy(src + key_dim, src + 2 * key_dim, &k[seq * key_dim]);
        std::copy(src + 2 * key_dim, src + qkv_dim, &v[seq * value_dim]);
    }

    normalize_l2_rows(q, seqlen * meta.linear_num_key_heads, meta.linear_key_head_dim, 1e-6f, meta.linear_key_head_dim == 0 ? 1.0f : std::pow(static_cast<float>(meta.linear_key_head_dim), -0.5f));
    normalize_l2_rows(k, seqlen * meta.linear_num_key_heads, meta.linear_key_head_dim, 1e-6f, 1.0f);

    std::vector<float> q_repeated = repeat_heads(
        q,
        seqlen,
        meta.linear_num_key_heads,
        meta.linear_num_value_heads,
        meta.linear_key_head_dim);
    std::vector<float> k_repeated = repeat_heads(
        k,
        seqlen,
        meta.linear_num_key_heads,
        meta.linear_num_value_heads,
        meta.linear_key_head_dim);

    const std::vector<float> &a_log = _linear_a_log_host.at(layer_idx);
    const std::vector<float> &dt_bias = _linear_dt_bias_host.at(layer_idx);
    std::vector<float> g(seqlen * meta.linear_num_value_heads, 0.0f);
    std::vector<float> beta(seqlen * meta.linear_num_value_heads, 0.0f);
    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t head = 0; head < meta.linear_num_value_heads; ++head) {
            const size_t idx = seq * meta.linear_num_value_heads + head;
            const float a_val = a_host[idx];
            const float b_val = b_host[idx];
            float softplus = 0.0f;
            const float shifted = a_val + dt_bias[head];
            if (shifted > 20.0f) {
                softplus = shifted;
            } else if (shifted < -20.0f) {
                softplus = std::exp(shifted);
            } else {
                softplus = std::log1p(std::exp(shifted));
            }
            g[idx] = -std::exp(a_log[head]) * softplus;
            beta[idx] = 1.0f / (1.0f + std::exp(-b_val));
        }
    }

    return {
        hostF32ToTensor(
            q_repeated,
            {seqlen, meta.linear_num_value_heads, meta.linear_key_head_dim},
            meta.dtype),
        hostF32ToTensor(
            k_repeated,
            {seqlen, meta.linear_num_value_heads, meta.linear_key_head_dim},
            meta.dtype),
        hostF32ToTensor(
            v,
            {seqlen, meta.linear_num_value_heads, meta.linear_value_head_dim},
            meta.dtype),
        hostF32ToTensor(
            z,
            {seqlen * meta.linear_num_value_heads, meta.linear_value_head_dim},
            meta.dtype),
        hostF32ToTensor(
            g,
            {seqlen, meta.linear_num_value_heads},
            meta.dtype),
        hostF32ToTensor(
            beta,
            {seqlen, meta.linear_num_value_heads},
            meta.dtype),
    };
}

tensor_t Model::prefillLinearAttentionLayer(tensor_t x, size_t layer_idx) {
    const size_t seqlen = x->shape()[0];
    const size_t value_dim = _meta.linear_num_value_heads * _meta.linear_value_head_dim;
    tensor_t x_norm = runAttentionInputNorm(x, layer_idx);
    LinearProjectedTensors projected = projectLinearAttentionInputs(x_norm, seqlen, layer_idx);

    const std::vector<float> qkv_host = tensorToHostF32(projected.qkv);
    const std::vector<float> z_host = tensorToHostF32(projected.z);
    const std::vector<float> a_host = tensorToHostF32(projected.a);
    const std::vector<float> b_host = tensorToHostF32(projected.b);
    const size_t qkv_dim = projected.qkv->shape()[1];
    const std::vector<float> mixed_qkv_host = host_linear_conv(
        qkv_host,
        seqlen,
        qkv_dim,
        _meta.linear_conv_kernel_dim,
        _linear_conv_host[layer_idx]);

    LinearPreparedTensors prepared = prepareLinearInputsFromHost(
        mixed_qkv_host,
        z_host,
        a_host,
        b_host,
        seqlen,
        layer_idx);

    tensor_t core_attn = makeTensor(
        {seqlen, _meta.linear_num_value_heads, _meta.linear_value_head_dim},
        _meta.dtype);
    tensor_t final_state = makeTensor(
        {_meta.linear_num_value_heads, _meta.linear_key_head_dim, _meta.linear_value_head_dim},
        _meta.dtype);
    ops::linear_attention(
        core_attn,
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        nullptr,
        final_state);

    const size_t keep = (_meta.linear_conv_kernel_dim > 0) ? (_meta.linear_conv_kernel_dim - 1) : 0;
    std::vector<float> &history = _linear_qkv_history[layer_idx];
    if (keep == 0) {
        history.clear();
    } else {
        const size_t rows_to_keep = std::min(keep, seqlen);
        history.assign(
            qkv_host.end() - static_cast<ptrdiff_t>(rows_to_keep * qkv_dim),
            qkv_host.end());
    }
    _linear_recurrent_state[layer_idx] = final_state;

    tensor_t core_attn_2d =
        core_attn->view({seqlen * _meta.linear_num_value_heads, _meta.linear_value_head_dim});
    tensor_t gated = makeTensor(
        {seqlen * _meta.linear_num_value_heads, _meta.linear_value_head_dim},
        _meta.dtype);
    ops::gated_rms_norm(
        gated,
        core_attn_2d,
        prepared.z,
        _weights.linear_attn_norm_w[layer_idx],
        _meta.epsilon);

    tensor_t gated_flat = gated->view({seqlen, value_dim});
    tensor_t attn_proj = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::linear(attn_proj, gated_flat, _weights.linear_attn_o_w[layer_idx], nullptr);
    tensor_t x_attn = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::add(x_attn, x, attn_proj);

    return x_attn;
}

tensor_t Model::decodeLinearAttentionLayer(tensor_t x, size_t layer_idx) {
    const size_t value_dim = _meta.linear_num_value_heads * _meta.linear_value_head_dim;
    tensor_t x_norm = runAttentionInputNorm(x, layer_idx);
    LinearProjectedTensors projected = projectLinearAttentionInputs(x_norm, 1, layer_idx);

    const std::vector<float> qkv_host = tensorToHostF32(projected.qkv);
    const std::vector<float> z_host = tensorToHostF32(projected.z);
    const std::vector<float> a_host = tensorToHostF32(projected.a);
    const std::vector<float> b_host = tensorToHostF32(projected.b);
    const size_t qkv_dim = projected.qkv->shape()[1];
    std::vector<float> combined_qkv = _linear_qkv_history[layer_idx];
    combined_qkv.insert(combined_qkv.end(), qkv_host.begin(), qkv_host.end());
    const size_t combined_rows = combined_qkv.size() / qkv_dim;
    const std::vector<float> mixed_qkv_host = host_linear_conv(
        combined_qkv,
        combined_rows,
        qkv_dim,
        _meta.linear_conv_kernel_dim,
        _linear_conv_host[layer_idx]);
    std::vector<float> last_mixed_qkv(
        mixed_qkv_host.end() - static_cast<ptrdiff_t>(qkv_dim),
        mixed_qkv_host.end());

    LinearPreparedTensors prepared = prepareLinearInputsFromHost(
        last_mixed_qkv,
        z_host,
        a_host,
        b_host,
        1,
        layer_idx);

    ASSERT(_linear_recurrent_state[layer_idx] != nullptr, "decodeLinearAttentionLayer: missing recurrent state");
    tensor_t final_state = makeTensor(
        {_meta.linear_num_value_heads, _meta.linear_key_head_dim, _meta.linear_value_head_dim},
        _meta.dtype);
    tensor_t core_attn = makeTensor(
        {1, _meta.linear_num_value_heads, _meta.linear_value_head_dim},
        _meta.dtype);
    ops::linear_attention(
        core_attn,
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        _linear_recurrent_state[layer_idx],
        final_state);

    const size_t keep = (_meta.linear_conv_kernel_dim > 0) ? (_meta.linear_conv_kernel_dim - 1) : 0;
    if (keep == 0) {
        _linear_qkv_history[layer_idx].clear();
    } else {
        const size_t rows_to_keep = std::min(keep, combined_rows);
        _linear_qkv_history[layer_idx].assign(
            combined_qkv.end() - static_cast<ptrdiff_t>(rows_to_keep * qkv_dim),
            combined_qkv.end());
    }
    _linear_recurrent_state[layer_idx] = final_state;

    tensor_t core_attn_2d =
        core_attn->view({_meta.linear_num_value_heads, _meta.linear_value_head_dim});
    tensor_t gated = makeTensor(
        {_meta.linear_num_value_heads, _meta.linear_value_head_dim},
        _meta.dtype);
    ops::gated_rms_norm(
        gated,
        core_attn_2d,
        prepared.z,
        _weights.linear_attn_norm_w[layer_idx],
        _meta.epsilon);

    tensor_t gated_flat = gated->view({1, value_dim});
    tensor_t attn_proj = makeTensor({1, _meta.hs}, _meta.dtype);
    ops::linear(attn_proj, gated_flat, _weights.linear_attn_o_w[layer_idx], nullptr);
    tensor_t x_attn = makeTensor({1, _meta.hs}, _meta.dtype);
    ops::add(x_attn, x, attn_proj);

    return x_attn;
}

tensor_t Model::prefillFullAttentionLayer(tensor_t x, size_t layer_idx, size_t start_pos) {
    const size_t seqlen = x->shape()[0];
    tensor_t x_norm = runAttentionInputNorm(x, layer_idx);
    FullProjectedTensors projected = projectFullAttentionInputs(x_norm, seqlen, layer_idx);

    const std::vector<float> q_proj_host = tensorToHostF32(projected.q_proj);
    const std::vector<float> k_proj_host = tensorToHostF32(projected.k_proj);
    const std::vector<float> v_proj_host = tensorToHostF32(projected.v_proj);
    FullAttentionPreparedTensors prepared = prepareFullAttentionInputsFromHost(
        q_proj_host,
        k_proj_host,
        v_proj_host,
        seqlen,
        layer_idx,
        start_pos);
    if (!_full_k_cache[layer_idx]) {
        _full_k_cache[layer_idx] = makeTensor({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype);
        _full_v_cache[layer_idx] = makeTensor({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype);
    }
    appendToCache(_full_k_cache[layer_idx], prepared.k, start_pos);
    appendToCache(_full_v_cache[layer_idx], prepared.v, start_pos);

    tensor_t attn_out = makeTensor({seqlen, _meta.nh, _meta.dh}, _meta.dtype);
    ops::self_attention(attn_out, prepared.q, prepared.k, prepared.v, std::pow(static_cast<float>(_meta.dh), -0.5f));
    tensor_t attn_flat = attn_out->view({seqlen, _meta.nh * _meta.dh});
    tensor_t gated_attn = applyFullAttentionGate(attn_flat, prepared.gate, seqlen);

    tensor_t attn_proj = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::linear(attn_proj, gated_attn, _weights.full_attn_o_w[layer_idx], nullptr);
    tensor_t x_attn = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::add(x_attn, x, attn_proj);

    return x_attn;
}

tensor_t Model::decodeFullAttentionLayer(tensor_t x, size_t layer_idx, size_t start_pos) {
    tensor_t x_norm = runAttentionInputNorm(x, layer_idx);
    FullProjectedTensors projected = projectFullAttentionInputs(x_norm, 1, layer_idx);

    const std::vector<float> q_proj_host = tensorToHostF32(projected.q_proj);
    const std::vector<float> k_proj_host = tensorToHostF32(projected.k_proj);
    const std::vector<float> v_proj_host = tensorToHostF32(projected.v_proj);
    FullAttentionPreparedTensors prepared = prepareFullAttentionInputsFromHost(
        q_proj_host,
        k_proj_host,
        v_proj_host,
        1,
        layer_idx,
        start_pos);
    ASSERT(_full_k_cache[layer_idx] != nullptr && _full_v_cache[layer_idx] != nullptr,
           "decodeFullAttentionLayer: missing full-attention cache");
    appendToCache(_full_k_cache[layer_idx], prepared.k, start_pos);
    appendToCache(_full_v_cache[layer_idx], prepared.v, start_pos);

    tensor_t k_ready = _full_k_cache[layer_idx]->slice(0, 0, start_pos + 1);
    tensor_t v_ready = _full_v_cache[layer_idx]->slice(0, 0, start_pos + 1);
    tensor_t attn_out = makeTensor({1, _meta.nh, _meta.dh}, _meta.dtype);
    ops::self_attention(attn_out, prepared.q, k_ready, v_ready, std::pow(static_cast<float>(_meta.dh), -0.5f));
    tensor_t attn_flat = attn_out->view({1, _meta.nh * _meta.dh});
    tensor_t gated_attn = applyFullAttentionGate(attn_flat, prepared.gate, 1);

    tensor_t attn_proj = makeTensor({1, _meta.hs}, _meta.dtype);
    ops::linear(attn_proj, gated_attn, _weights.full_attn_o_w[layer_idx], nullptr);
    tensor_t x_attn = makeTensor({1, _meta.hs}, _meta.dtype);
    ops::add(x_attn, x, attn_proj);

    return x_attn;
}

tensor_t Model::finalLogitsFromHidden(tensor_t hidden, size_t seqlen) const {
    tensor_t x_norm = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::rms_norm(x_norm, hidden, _weights.out_norm_w, _meta.epsilon);
    tensor_t logits = makeTensor({seqlen, _meta.voc}, _meta.dtype);
    tensor_t out_embed = _weights.out_embed ? _weights.out_embed : _weights.in_embed;
    ops::linear(logits, x_norm, out_embed, nullptr);
    return logits;
}

tensor_t Model::applyFullAttentionGate(tensor_t attn_flat, const std::vector<float> &gate, size_t seqlen) const {
    std::vector<float> attn_flat_host = tensorToHostF32(attn_flat);
    ASSERT(attn_flat_host.size() == gate.size(), "applyFullAttentionGate: gate size mismatch");
    for (size_t i = 0; i < attn_flat_host.size(); ++i) {
        attn_flat_host[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
    }
    return hostF32ToTensor(attn_flat_host, {seqlen, _meta.nh * _meta.dh}, _meta.dtype);
}

tensor_t Model::runMLPBlock(tensor_t x, size_t layer_idx) const {
    const size_t seqlen = x->shape()[0];
    tensor_t x_post = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::rms_norm(x_post, x, _weights.mlp_norm_w[layer_idx], _meta.epsilon);

    tensor_t gate_proj = makeTensor({seqlen, _meta.di}, _meta.dtype);
    tensor_t up_proj = makeTensor({seqlen, _meta.di}, _meta.dtype);
    ops::linear(gate_proj, x_post, _weights.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up_proj, x_post, _weights.mlp_up_w[layer_idx], nullptr);

    tensor_t swiglu = makeTensor({seqlen, _meta.di}, _meta.dtype);
    ops::swiglu(swiglu, gate_proj, up_proj);
    tensor_t mlp_out = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::linear(mlp_out, swiglu, _weights.mlp_down_w[layer_idx], nullptr);

    tensor_t x_mlp = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::add(x_mlp, x, mlp_out);
    return x_mlp;
}

tensor_t Model::prefillLogits(const int64_t *token_ids, size_t ntoken) {
    tensor_t input_ids = makeTensor({ntoken}, WGINFER_DTYPE_I64);
    input_ids->load(token_ids);
    tensor_t x = makeTensor({ntoken, _meta.hs}, _meta.dtype);
    ops::embedding(x, input_ids, _weights.in_embed);
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        x = prefillDecoderLayer(x, i, 0);
    }
    return finalLogitsFromHidden(x, ntoken);
}

tensor_t Model::prefillDecoderLayer(tensor_t x, size_t layer_idx, size_t start_pos) {
    tensor_t attn_out;
    if (_layer_types[layer_idx] == LayerType::LinearAttention) {
        attn_out = prefillLinearAttentionLayer(x, layer_idx);
    } else {
        attn_out = prefillFullAttentionLayer(x, layer_idx, start_pos);
    }
    return runMLPBlock(attn_out, layer_idx);
}

tensor_t Model::decodeOneTokenLogits(int64_t token_id) {
    tensor_t input_ids = makeTensor({1}, WGINFER_DTYPE_I64);
    input_ids->load(&token_id);
    tensor_t x = makeTensor({1, _meta.hs}, _meta.dtype);
    ops::embedding(x, input_ids, _weights.in_embed);
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        x = decodeDecoderLayer(x, i, _cache_len);
    }
    return finalLogitsFromHidden(x, 1);
}

tensor_t Model::decodeDecoderLayer(tensor_t x, size_t layer_idx, size_t start_pos) {
    tensor_t attn_out;
    if (_layer_types[layer_idx] == LayerType::LinearAttention) {
        attn_out = decodeLinearAttentionLayer(x, layer_idx);
    } else {
        attn_out = decodeFullAttentionLayer(x, layer_idx, start_pos);
    }
    return runMLPBlock(attn_out, layer_idx);
}

int64_t Model::greedyNextToken(tensor_t logits) const {
    tensor_t last_logits = logits->slice(0, logits->shape()[0] - 1, logits->shape()[0])->view({_meta.voc});
    tensor_t max_idx = makeTensor({1}, WGINFER_DTYPE_I64);
    tensor_t max_val = makeTensor({1}, _meta.dtype);
    ops::argmax(max_idx, max_val, last_logits);

    int64_t host_result = 0;
    wginfer::core::context().setDevice(_device_type, _device_id);
    const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();
    api->memcpy_sync(
        &host_result,
        max_idx->data(),
        sizeof(int64_t),
        (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2H);
    return host_result;
}

int64_t Model::infer(
    int64_t *token_ids,
    size_t ntoken,
    int top_k,
    float top_p,
    float temperature) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen3.5 infer: token_ids must not be null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen3.5 infer: ntoken must be positive");
    wginfer::core::context().setDevice(_device_type, _device_id);

    tensor_t logits;
    if (_cache_len == 0) {
        logits = prefillLogits(token_ids, ntoken);
        _cache_len = ntoken;
    } else if (ntoken == 1) {
        logits = decodeOneTokenLogits(token_ids[0]);
        _cache_len += 1;
    } else {
        for (size_t i = 0; i < ntoken; ++i) {
            logits = decodeOneTokenLogits(token_ids[i]);
            _cache_len += 1;
        }
    }

    const bool greedy =
        (top_k == 1) &&
        (top_p >= 1.0f) &&
        (std::abs(temperature - 1.0f) < 1e-6f);
    if (greedy) {
        return greedyNextToken(logits);
    }

    tensor_t last_logits = logits->slice(0, logits->shape()[0] - 1, logits->shape()[0])->view({_meta.voc});
    return sample_from_logits(tensorToHostF32(last_logits), top_k, top_p, temperature);
}

} // namespace wginfer::models::qwen3_5
