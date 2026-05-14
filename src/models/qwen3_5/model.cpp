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
      _linear_qkv_history_gpu(meta.nlayer),
      _linear_qkv_history_rows(meta.nlayer, 0) {
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
}

void Model::reset_cache() {
    _cache_len = 0;
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _linear_qkv_history_gpu[i] = nullptr;
        _linear_qkv_history_rows[i] = 0;
        _linear_recurrent_state[i] = nullptr;
    }
}

Model::~Model() {
    reset_cache();
    for (auto &t : _full_k_cache) t = nullptr;
    for (auto &t : _full_v_cache) t = nullptr;
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

LinearPreparedTensors Model::prepareLinearInputs(
    tensor_t mixed_qkv,
    tensor_t z,
    tensor_t a,
    tensor_t b,
    size_t seqlen,
    size_t layer_idx) const {
    const ModelMeta &meta = _meta;
    tensor_t q = makeTensor({seqlen, meta.linear_num_value_heads, meta.linear_key_head_dim}, meta.dtype);
    tensor_t k = makeTensor({seqlen, meta.linear_num_value_heads, meta.linear_key_head_dim}, meta.dtype);
    tensor_t v = makeTensor({seqlen, meta.linear_num_value_heads, meta.linear_value_head_dim}, meta.dtype);
    const float q_scale = meta.linear_key_head_dim == 0
        ? 1.0f
        : std::pow(static_cast<float>(meta.linear_key_head_dim), -0.5f);
    ops::linear_attn_qkv_prepare(
        q,
        k,
        v,
        mixed_qkv,
        meta.linear_num_key_heads,
        1e-6f,
        q_scale);

    tensor_t g = makeTensor({seqlen, meta.linear_num_value_heads}, WGINFER_DTYPE_F32);
    tensor_t beta = makeTensor({seqlen, meta.linear_num_value_heads}, WGINFER_DTYPE_F32);
    ops::linear_attn_gates(
        g,
        beta,
        a,
        b,
        _weights.linear_attn_a_log[layer_idx],
        _weights.linear_attn_dt_bias[layer_idx]);

    return {
        q,
        k,
        v,
        z->view({seqlen * meta.linear_num_value_heads, meta.linear_value_head_dim}),
        g,
        beta,
    };
}

tensor_t Model::prefillLinearAttentionLayer(tensor_t x, size_t layer_idx) {
    const size_t seqlen = x->shape()[0];
    const size_t value_dim = _meta.linear_num_value_heads * _meta.linear_value_head_dim;
    tensor_t x_norm = runAttentionInputNorm(x, layer_idx);
    LinearProjectedTensors projected = projectLinearAttentionInputs(x_norm, seqlen, layer_idx);

    const size_t qkv_dim = projected.qkv->shape()[1];
    const size_t elem_size = projected.qkv->elementSize();
    tensor_t mixed_qkv = makeTensor({seqlen, qkv_dim}, _meta.dtype);
    ops::causal_conv1d(mixed_qkv, projected.qkv, _weights.linear_attn_conv_w[layer_idx]);

    LinearPreparedTensors prepared = prepareLinearInputs(
        mixed_qkv,
        projected.z,
        projected.a,
        projected.b,
        seqlen,
        layer_idx);

    tensor_t core_attn = makeTensor(
        {seqlen, _meta.linear_num_value_heads, _meta.linear_value_head_dim},
        _meta.dtype);
    tensor_t final_state = makeTensor(
        {_meta.linear_num_value_heads, _meta.linear_key_head_dim, _meta.linear_value_head_dim},
        WGINFER_DTYPE_F32);
    ops::linear_attention(
        core_attn,
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        nullptr,
        final_state);

    // Save last K-1 rows of QKV on GPU for decode
    const size_t keep = (_meta.linear_conv_kernel_dim > 0) ? (_meta.linear_conv_kernel_dim - 1) : 0;
    if (keep > 0) {
        const size_t rows_to_keep = std::min(keep, seqlen);
        tensor_t history_slice = projected.qkv->slice(0, seqlen - rows_to_keep, seqlen);
        if (!_linear_qkv_history_gpu[layer_idx] ||
            _linear_qkv_history_gpu[layer_idx]->shape()[0] != rows_to_keep) {
            _linear_qkv_history_gpu[layer_idx] = makeTensor({rows_to_keep, qkv_dim}, _meta.dtype);
        }
        wginfer::core::context().setDevice(_device_type, _device_id);
        const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();
        api->memcpy_sync(
            _linear_qkv_history_gpu[layer_idx]->data(),
            history_slice->data(),
            history_slice->numel() * elem_size,
            (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2D);
        _linear_qkv_history_rows[layer_idx] = rows_to_keep;
    } else {
        _linear_qkv_history_gpu[layer_idx] = nullptr;
        _linear_qkv_history_rows[layer_idx] = 0;
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

    const size_t qkv_dim = projected.qkv->shape()[1];
    const size_t elem_size = projected.qkv->elementSize();
    const size_t keep = (_meta.linear_conv_kernel_dim > 0) ? (_meta.linear_conv_kernel_dim - 1) : 0;

    // Build combined QKV tensor on device: [history_rows + 1, qkv_dim]
    const size_t history_rows = _linear_qkv_history_rows[layer_idx];
    const size_t total_rows = history_rows + 1;
    tensor_t combined_qkv = makeTensor({total_rows, qkv_dim}, _meta.dtype);

    wginfer::core::context().setDevice(_device_type, _device_id);
    const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();
    const wginferMemcpyKind_t copy_kind =
        (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2D;

    // Copy history to beginning of combined buffer
    if (history_rows > 0 && _linear_qkv_history_gpu[layer_idx]) {
        api->memcpy_sync(
            combined_qkv->data(),
            _linear_qkv_history_gpu[layer_idx]->data(),
            history_rows * qkv_dim * elem_size,
            copy_kind);
    }
    // Copy current qkv to last slot
    api->memcpy_sync(
        static_cast<std::byte *>(combined_qkv->data()) + history_rows * qkv_dim * elem_size,
        projected.qkv->data(),
        qkv_dim * elem_size,
        copy_kind);

    // Run causal_conv1d on the combined buffer and take last output
    tensor_t mixed_qkv = makeTensor({total_rows, qkv_dim}, _meta.dtype);
    ops::causal_conv1d(mixed_qkv, combined_qkv, _weights.linear_attn_conv_w[layer_idx]);
    tensor_t last_mixed_qkv = mixed_qkv->slice(0, total_rows - 1, total_rows);

    LinearPreparedTensors prepared = prepareLinearInputs(
        last_mixed_qkv,
        projected.z,
        projected.a,
        projected.b,
        1,
        layer_idx);

    ASSERT(_linear_recurrent_state[layer_idx] != nullptr, "decodeLinearAttentionLayer: missing recurrent state");
    tensor_t final_state = makeTensor(
        {_meta.linear_num_value_heads, _meta.linear_key_head_dim, _meta.linear_value_head_dim},
        WGINFER_DTYPE_F32);
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

    // Update GPU history: save last K-1 rows of input qkv
    if (keep > 0) {
        const size_t rows_to_keep = std::min(keep, total_rows);
        tensor_t src_slice = combined_qkv->slice(0, total_rows - rows_to_keep, total_rows);
        if (!_linear_qkv_history_gpu[layer_idx] ||
            _linear_qkv_history_gpu[layer_idx]->shape()[0] != rows_to_keep) {
            _linear_qkv_history_gpu[layer_idx] = makeTensor({rows_to_keep, qkv_dim}, _meta.dtype);
        }
        api->memcpy_sync(
            _linear_qkv_history_gpu[layer_idx]->data(),
            src_slice->data(),
            rows_to_keep * qkv_dim * elem_size,
            copy_kind);
        _linear_qkv_history_rows[layer_idx] = rows_to_keep;
    } else {
        _linear_qkv_history_gpu[layer_idx] = nullptr;
        _linear_qkv_history_rows[layer_idx] = 0;
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

    // q_proj layout: [seqlen, nh * dh * 2] with interleaved [Q_per_head, gate_per_head]
    // Split on host (data is small: ~86KB for 14 tokens), then norm/rope on GPU
    const std::vector<float> q_proj_host = tensorToHostF32(projected.q_proj);
    const size_t nh_dh = _meta.nh * _meta.dh;
    std::vector<float> q_host(seqlen * nh_dh), gate_host(seqlen * nh_dh);
    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t h = 0; h < _meta.nh; ++h) {
            const size_t src_off = (seq * _meta.nh + h) * _meta.dh * 2;
            const size_t dst_off = (seq * _meta.nh + h) * _meta.dh;
            std::copy_n(q_proj_host.data() + src_off, _meta.dh, q_host.data() + dst_off);
            std::copy_n(q_proj_host.data() + src_off + _meta.dh, _meta.dh, gate_host.data() + dst_off);
        }
    }
    tensor_t q_raw = hostF32ToTensor(q_host, {seqlen * _meta.nh, _meta.dh}, _meta.dtype);
    tensor_t gate = hostF32ToTensor(gate_host, {seqlen, _meta.nh * _meta.dh}, _meta.dtype);

    // Q norm on GPU
    tensor_t q_norm = makeTensor({seqlen * _meta.nh, _meta.dh}, _meta.dtype);
    ops::rms_norm(q_norm, q_raw, _weights.full_attn_q_norm_w[layer_idx], _meta.epsilon);

    // K norm on GPU (k_proj is already contiguous)
    tensor_t k_norm = makeTensor({seqlen * _meta.nkvh, _meta.dh}, _meta.dtype);
    ops::rms_norm(k_norm, projected.k_proj->view({seqlen * _meta.nkvh, _meta.dh}),
                  _weights.full_attn_k_norm_w[layer_idx], _meta.epsilon);

    // V: raw projection, just reshape
    tensor_t v = projected.v_proj->view({seqlen, _meta.nkvh, _meta.dh});

    // Partial RoPE on GPU
    const size_t rotary_dim = static_cast<size_t>(_meta.dh * _meta.partial_rotary_factor) & ~1ULL;
    tensor_t q = q_norm->view({seqlen, _meta.nh, _meta.dh});
    tensor_t k = k_norm->view({seqlen, _meta.nkvh, _meta.dh});

    if (rotary_dim > 0) {
        tensor_t pos_ids = makeTensor({seqlen}, WGINFER_DTYPE_I64);
        {
            std::vector<int64_t> host_pos(seqlen);
            for (size_t i = 0; i < seqlen; ++i) host_pos[i] = static_cast<int64_t>(start_pos + i);
            pos_ids->load(host_pos.data());
        }
        tensor_t q_rope = makeTensor({seqlen, _meta.nh, _meta.dh}, _meta.dtype);
        tensor_t k_rope = makeTensor({seqlen, _meta.nkvh, _meta.dh}, _meta.dtype);
        ops::rope_partial(q_rope, q, pos_ids, _meta.theta, rotary_dim);
        ops::rope_partial(k_rope, k, pos_ids, _meta.theta, rotary_dim);
        q = q_rope;
        k = k_rope;
    }

    // KV cache
    if (!_full_k_cache[layer_idx]) {
        _full_k_cache[layer_idx] = makeTensor({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype);
        _full_v_cache[layer_idx] = makeTensor({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype);
    }
    appendToCache(_full_k_cache[layer_idx], k, start_pos);
    appendToCache(_full_v_cache[layer_idx], v, start_pos);

    // Self attention
    tensor_t attn_out = makeTensor({seqlen, _meta.nh, _meta.dh}, _meta.dtype);
    ops::self_attention(attn_out, q, k, v, std::pow(static_cast<float>(_meta.dh), -0.5f));

    // Gate (mul_sigmoid)
    tensor_t attn_flat = attn_out->view({seqlen, _meta.nh * _meta.dh});
    tensor_t gated_attn = applyFullAttentionGate(attn_flat, gate, seqlen);

    // Output projection + residual
    tensor_t attn_proj = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::linear(attn_proj, gated_attn, _weights.full_attn_o_w[layer_idx], nullptr);
    tensor_t x_attn = makeTensor({seqlen, _meta.hs}, _meta.dtype);
    ops::add(x_attn, x, attn_proj);

    return x_attn;
}

tensor_t Model::decodeFullAttentionLayer(tensor_t x, size_t layer_idx, size_t start_pos) {
    tensor_t x_norm = runAttentionInputNorm(x, layer_idx);
    FullProjectedTensors projected = projectFullAttentionInputs(x_norm, 1, layer_idx);

    // Split Q and gate on host (tiny data: 1 token)
    const std::vector<float> q_proj_host = tensorToHostF32(projected.q_proj);
    const size_t nh_dh = _meta.nh * _meta.dh;
    std::vector<float> q_host(nh_dh), gate_host(nh_dh);
    for (size_t h = 0; h < _meta.nh; ++h) {
        std::copy_n(q_proj_host.data() + h * _meta.dh * 2, _meta.dh, q_host.data() + h * _meta.dh);
        std::copy_n(q_proj_host.data() + h * _meta.dh * 2 + _meta.dh, _meta.dh, gate_host.data() + h * _meta.dh);
    }
    tensor_t q_raw = hostF32ToTensor(q_host, {_meta.nh, _meta.dh}, _meta.dtype);
    tensor_t gate = hostF32ToTensor(gate_host, {1, _meta.nh * _meta.dh}, _meta.dtype);

    // Q norm on GPU
    tensor_t q_norm = makeTensor({_meta.nh, _meta.dh}, _meta.dtype);
    ops::rms_norm(q_norm, q_raw, _weights.full_attn_q_norm_w[layer_idx], _meta.epsilon);
    tensor_t q = q_norm->view({1, _meta.nh, _meta.dh});

    // K norm on GPU
    tensor_t k_norm = makeTensor({_meta.nkvh, _meta.dh}, _meta.dtype);
    ops::rms_norm(k_norm, projected.k_proj->view({_meta.nkvh, _meta.dh}),
                  _weights.full_attn_k_norm_w[layer_idx], _meta.epsilon);
    tensor_t k = k_norm->view({1, _meta.nkvh, _meta.dh});

    // V: raw
    tensor_t v = projected.v_proj->view({1, _meta.nkvh, _meta.dh});

    // Partial RoPE on GPU
    const size_t rotary_dim = static_cast<size_t>(_meta.dh * _meta.partial_rotary_factor) & ~1ULL;
    if (rotary_dim > 0) {
        tensor_t pos_ids = makeTensor({1}, WGINFER_DTYPE_I64);
        int64_t host_pos = static_cast<int64_t>(start_pos);
        pos_ids->load(&host_pos);
        tensor_t q_rope = makeTensor({1, _meta.nh, _meta.dh}, _meta.dtype);
        tensor_t k_rope = makeTensor({1, _meta.nkvh, _meta.dh}, _meta.dtype);
        ops::rope_partial(q_rope, q, pos_ids, _meta.theta, rotary_dim);
        ops::rope_partial(k_rope, k, pos_ids, _meta.theta, rotary_dim);
        q = q_rope;
        k = k_rope;
    }

    // KV cache
    ASSERT(_full_k_cache[layer_idx] != nullptr && _full_v_cache[layer_idx] != nullptr,
           "decodeFullAttentionLayer: missing full-attention cache");
    appendToCache(_full_k_cache[layer_idx], k, start_pos);
    appendToCache(_full_v_cache[layer_idx], v, start_pos);

    // Self attention with full cache
    tensor_t k_ready = _full_k_cache[layer_idx]->slice(0, 0, start_pos + 1);
    tensor_t v_ready = _full_v_cache[layer_idx]->slice(0, 0, start_pos + 1);
    tensor_t attn_out = makeTensor({1, _meta.nh, _meta.dh}, _meta.dtype);
    ops::self_attention(attn_out, q, k_ready, v_ready, std::pow(static_cast<float>(_meta.dh), -0.5f));

    // Gate
    tensor_t attn_flat = attn_out->view({1, _meta.nh * _meta.dh});
    tensor_t gated_attn = applyFullAttentionGate(attn_flat, gate, 1);

    // Output projection + residual
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

tensor_t Model::applyFullAttentionGate(tensor_t attn_flat, tensor_t gate, size_t seqlen) const {
    tensor_t gated = makeTensor({seqlen, _meta.nh * _meta.dh}, _meta.dtype);
    ops::mul_sigmoid(gated, attn_flat, gate);
    return gated;
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
