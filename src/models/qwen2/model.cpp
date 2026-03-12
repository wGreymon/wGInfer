#include "model.hpp"
#include "../../core/wginfer_core.hpp"
#include "../../device/runtime_api.hpp"
#include "../../ops/add/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace wginfer::models::qwen2 {

namespace {

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

std::vector<float> logits_to_host_f32(tensor_t logits, const WginferRuntimeAPI *api) {
    const size_t n = logits->numel();
    std::vector<float> out(n);
    switch (logits->dtype()) {
    case WGINFER_DTYPE_F32: {
        api->memcpy_sync(out.data(), logits->data(), n * sizeof(float), WGINFER_MEMCPY_D2H);
        break;
    }
    case WGINFER_DTYPE_F16: {
        std::vector<wginfer::fp16_t> tmp(n);
        api->memcpy_sync(tmp.data(), logits->data(), n * sizeof(wginfer::fp16_t), WGINFER_MEMCPY_D2H);
        for (size_t i = 0; i < n; ++i) {
            out[i] = wginfer::utils::cast<float>(tmp[i]);
        }
        break;
    }
    case WGINFER_DTYPE_BF16: {
        std::vector<wginfer::bf16_t> tmp(n);
        api->memcpy_sync(tmp.data(), logits->data(), n * sizeof(wginfer::bf16_t), WGINFER_MEMCPY_D2H);
        for (size_t i = 0; i < n; ++i) {
            out[i] = wginfer::utils::cast<float>(tmp[i]);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits->dtype());
    }
    return out;
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

    const size_t vocab = logits.size();
    if (top_k <= 0 || top_k > static_cast<int>(vocab)) {
        top_k = static_cast<int>(vocab);
    }
    if (top_p <= 0.0f || top_p > 1.0f) {
        top_p = 1.0f;
    }

    if (top_k == 1 && top_p >= 1.0f) {
        return argmax_host(logits);
    }

    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    auto by_logit_desc = [&logits](int a, int b) { return logits[a] > logits[b]; };
    // 对idx根据logit的值进行排序，a和b是idx中的元素
    if (top_k < static_cast<int>(vocab)) {
        //  [first, last) 范围里挑选出“应该排在最前面的那一批元素”，把它们放进 [first, middle)，
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), by_logit_desc);
        idx.resize(top_k);
    } else {
        std::sort(idx.begin(), idx.end(), by_logit_desc);
    }

    // temperature < 1时，分数差距被放大，分布更尖；>1时，分数差距被缩小，分布更平
    const float inv_temp = 1.0f / temperature;
    float max_scaled = -std::numeric_limits<float>::infinity();
    for (int i : idx) {
        max_scaled = std::max(max_scaled, logits[i] * inv_temp);
    }

    // 用softmax将logits[i]从原始分数映射成可采样权重
    std::vector<double> probs(idx.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        double p = std::exp(static_cast<double>(logits[idx[i]] * inv_temp - max_scaled));
        if (!std::isfinite(p) || p < 0.0) {
            p = 0.0;
        }
        probs[i] = p;
        total += p;
    }
    if (total <= 0.0) {
        return static_cast<int64_t>(idx.front());
    }

    if (top_p < 1.0f) {
        double cum = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i] / total;
            keep = i + 1;
            if (cum >= static_cast<double>(top_p)) {
                break;
            }
        }
        keep = std::max<size_t>(keep, 1);
        idx.resize(keep);
        probs.resize(keep);
    }

    thread_local std::mt19937 rng(std::random_device{}());
    // 根据一组权重，按概率随机返回某个整数下标。
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int chosen = dist(rng);
    return static_cast<int64_t>(idx[static_cast<size_t>(chosen)]);
}

} // namespace

Model::Model(const ModelMeta &meta, wginferDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id), _cache_len(0) {

    // 初始化 KV Cache，每层都有KV Cache
    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _k_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh},
                                     _meta.dtype, _device_type, _device_id);
        _v_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh},
                                     _meta.dtype, _device_type, _device_id);
    }

    // 初始化权重数组
    _weights.attn_norm_w.resize(_meta.nlayer);
    _weights.attn_q_w.resize(_meta.nlayer);
    _weights.attn_q_b.resize(_meta.nlayer);
    _weights.attn_k_w.resize(_meta.nlayer);
    _weights.attn_k_b.resize(_meta.nlayer);
    _weights.attn_v_w.resize(_meta.nlayer);
    _weights.attn_v_b.resize(_meta.nlayer);
    _weights.attn_o_w.resize(_meta.nlayer);
    _weights.mlp_norm_w.resize(_meta.nlayer);
    _weights.mlp_gate_w.resize(_meta.nlayer);
    _weights.mlp_up_w.resize(_meta.nlayer);
    _weights.mlp_down_w.resize(_meta.nlayer);

    // 创建 dummy bias tensors（全零，用于没有 bias 的层）
    _dummy_bias_hs = Tensor::create({_meta.hs}, _meta.dtype, _device_type, _device_id);
    _dummy_bias_di = Tensor::create({_meta.di}, _meta.dtype, _device_type, _device_id);
    _dummy_bias_q = Tensor::create({_meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
    _dummy_bias_kv = Tensor::create({_meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
    _dummy_bias_voc = Tensor::create({_meta.voc}, _meta.dtype, _device_type, _device_id);

    // dummy bias 必须显式清零，否则会把未初始化内存当作 bias 加进去，导致输出完全错误
    auto zero_tensor = [](const tensor_t &t) {
        std::vector<std::byte> zeros(t->numel() * t->elementSize(), std::byte{0});
        t->load(zeros.data());
    };
    zero_tensor(_dummy_bias_hs);
    zero_tensor(_dummy_bias_di);
    zero_tensor(_dummy_bias_q);
    zero_tensor(_dummy_bias_kv);
    zero_tensor(_dummy_bias_voc);
}

Model::~Model() {
    // 智能指针会自动管理内存
}

void Model::reset_cache() {
    _cache_len = 0;
}

// 有就复用，没有就创建
void Model::ensure_tensor(tensor_t &tensor, const std::vector<size_t> &shape, wginferDataType_t dtype) {
    const bool need_new = (!tensor)
                       || tensor->dtype() != dtype
                       || tensor->deviceType() != _device_type
                       || tensor->deviceId() != _device_id
                       || tensor->shape() != shape;
    if (need_new) {
        tensor = Tensor::create(shape, dtype, _device_type, _device_id);
    }
}

void Model::update_kv_cache(size_t layer_idx, tensor_t k_new, tensor_t v_new, size_t seqlen, size_t old_len) {
    // 将新的 K 和 V 追加到 cache
    // k_new: [seqlen, nkvh, dh]
    // v_new: [seqlen, nkvh, dh]

    // old_len 必须是"本次 forward 开始前"的 cache 长度。
    // 注意：_cache_len 是全局序列长度，不应在每一层里自增。
    ASSERT(old_len == _cache_len, "update_kv_cache: old_len must equal _cache_len");
    size_t new_len = old_len + seqlen;
    CHECK_ARGUMENT(new_len <= _meta.maxseq, "update_kv_cache: cache overflow");

    // 复制新计算的 K 和 V 到 cache
    // 使用运行时 API 的内存拷贝，支持跨设备
    wginfer::core::context().setDevice(_device_type, _device_id);
    const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();

    // 使用 tensor 的 numel 和 elementSize 计算正确的字节数
    size_t k_size = k_new->numel() * k_new->elementSize();
    size_t v_size = v_new->numel() * v_new->elementSize();

    // 确保 k_new 和 v_new 是连续的
    ASSERT(k_new->isContiguous() && v_new->isContiguous(),
           "update_kv_cache: k_new and v_new must be contiguous");
    ASSERT(_k_cache[layer_idx]->isContiguous() && _v_cache[layer_idx]->isContiguous(),
           "update_kv_cache: cache tensors must be contiguous");

    // cache/new 都在同一设备上；CPU 用 H2H，其它设备用 D2D。
    const size_t cache_row_bytes = _meta.nkvh * _meta.dh * k_new->elementSize();
    const size_t dst_offset_bytes = old_len * cache_row_bytes;
    const wginferMemcpyKind_t copy_kind =
        (_device_type == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_D2D;
    api->memcpy_sync(_k_cache[layer_idx]->data() + dst_offset_bytes, k_new->data(), k_size, copy_kind);
    api->memcpy_sync(_v_cache[layer_idx]->data() + dst_offset_bytes, v_new->data(), v_size, copy_kind);
}

// transformer block
void Model::forward_layer(size_t layer_idx, tensor_t &x, size_t seqlen, size_t total_len, tensor_t pos_ids_q) {
    // 设置设备上下文
    wginfer::core::context().setDevice(_device_type, _device_id);

    // 1. Pre-attention norm
    ensure_tensor(_x_norm, {seqlen, _meta.hs}, _meta.dtype);
    ops::rms_norm(_x_norm, x, _weights.attn_norm_w[layer_idx], _meta.epsilon);

    // 2. Attention
    // 2.1 计算 Q, K, V
    // x_norm: [seqlen, hs]
    // Q weight: [nh * dh, hs], output: [seqlen, nh * dh]
    // K weight: [nkvh * dh, hs], output: [seqlen, nkvh * dh]
    // V weight: [nkvh * dh, hs], output: [seqlen, nkvh * dh]

    ensure_tensor(_q_flat, {seqlen, _meta.nh * _meta.dh}, _meta.dtype);
    ensure_tensor(_k_flat, {seqlen, _meta.nkvh * _meta.dh}, _meta.dtype);
    ensure_tensor(_v_flat, {seqlen, _meta.nkvh * _meta.dh}, _meta.dtype);

    // 处理可能为空的 bias：如果不存在，使用 dummy bias
    tensor_t q_bias = (_weights.attn_q_b[layer_idx] && _weights.attn_q_b[layer_idx]->numel() > 0) ? _weights.attn_q_b[layer_idx] : _dummy_bias_q;
    tensor_t k_bias = (_weights.attn_k_b[layer_idx] && _weights.attn_k_b[layer_idx]->numel() > 0) ? _weights.attn_k_b[layer_idx] : _dummy_bias_kv;
    tensor_t v_bias = (_weights.attn_v_b[layer_idx] && _weights.attn_v_b[layer_idx]->numel() > 0) ? _weights.attn_v_b[layer_idx] : _dummy_bias_kv;

    ops::linear(_q_flat, _x_norm, _weights.attn_q_w[layer_idx], q_bias);
    ops::linear(_k_flat, _x_norm, _weights.attn_k_w[layer_idx], k_bias);
    ops::linear(_v_flat, _x_norm, _weights.attn_v_w[layer_idx], v_bias);

    // Reshape: [seqlen, nh * dh] -> [seqlen, nh, dh]
    _q = _q_flat->view({seqlen, _meta.nh, _meta.dh});
    _k = _k_flat->view({seqlen, _meta.nkvh, _meta.dh});
    _v = _v_flat->view({seqlen, _meta.nkvh, _meta.dh});

    // 2.2 RoPE（只处理本轮新增 token）
    ensure_tensor(_q_rope, {seqlen, _meta.nh, _meta.dh}, _meta.dtype);
    ensure_tensor(_k_rope_new, {seqlen, _meta.nkvh, _meta.dh}, _meta.dtype);
    ops::rope(_k_rope_new, _k, pos_ids_q, _meta.theta);
    ops::rope(_q_rope, _q, pos_ids_q, _meta.theta);

    // 2.3 更新 KV Cache（K 使用 RoPE 后结果，V 保持原值）
    size_t old_len = total_len - seqlen;
    update_kv_cache(layer_idx, _k_rope_new, _v, seqlen, old_len);

    // 2.4 准备完整的 K 和 V（包含 cache）
    _k_full = _k_cache[layer_idx]->slice(0, 0, total_len);
    _v_full = _v_cache[layer_idx]->slice(0, 0, total_len);

    // 2.5 Self-attention
    ensure_tensor(_attn_out, {seqlen, _meta.nh, _meta.dh}, _meta.dtype);
    float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    ops::self_attention(_attn_out, _q_rope, _k_full, _v_full, scale);

    // 2.6 Attention output projection
    // attn_out: [seqlen, nh, dh] -> [seqlen, nh * dh]
    tensor_t attn_out_flat = _attn_out->view({seqlen, _meta.nh * _meta.dh});
    ensure_tensor(_attn_proj_out, {seqlen, _meta.hs}, _meta.dtype);
    ops::linear(_attn_proj_out, attn_out_flat, _weights.attn_o_w[layer_idx], nullptr);

    // 2.7 残差连接
    ensure_tensor(_x_attn, {seqlen, _meta.hs}, _meta.dtype);
    ops::add(_x_attn, x, _attn_proj_out);
    x = _x_attn;

    // 3. Post-attention norm
    ensure_tensor(_x_norm, {seqlen, _meta.hs}, _meta.dtype);
    ops::rms_norm(_x_norm, x, _weights.mlp_norm_w[layer_idx], _meta.epsilon);

    // 4. MLP
    // x_norm: [seqlen, hs]
    ensure_tensor(_gate, {seqlen, _meta.di}, _meta.dtype);
    ensure_tensor(_up, {seqlen, _meta.di}, _meta.dtype);

    ops::linear(_gate, _x_norm, _weights.mlp_gate_w[layer_idx], nullptr);
    ops::linear(_up, _x_norm, _weights.mlp_up_w[layer_idx], nullptr);

    ensure_tensor(_swiglu_out, {seqlen, _meta.di}, _meta.dtype);
    ops::swiglu(_swiglu_out, _gate, _up);

    ensure_tensor(_mlp_out, {seqlen, _meta.hs}, _meta.dtype);
    ops::linear(_mlp_out, _swiglu_out, _weights.mlp_down_w[layer_idx], nullptr);

    // 5. 残差连接
    ensure_tensor(_x_mlp, {seqlen, _meta.hs}, _meta.dtype);
    ops::add(_x_mlp, x, _mlp_out);
    x = _x_mlp;
}

// 送入一批token_ids, 输出每个位置对整个词表的logits
tensor_t Model::forward(tensor_t input_ids, size_t seqlen, size_t total_len) {
    // 设置设备上下文
    wginfer::core::context().setDevice(_device_type, _device_id);

    // 1. Embedding
    ensure_tensor(_x, {seqlen, _meta.hs}, _meta.dtype);
    ops::embedding(_x, input_ids, _weights.in_embed);

    // 2. 本轮所有层复用同一份 pos_ids（避免每层重复构造与拷贝）
    size_t start_pos = total_len - seqlen;
    ensure_tensor(_pos_ids_q, {seqlen}, WGINFER_DTYPE_I64);
    if (seqlen == 1) {
        int64_t pos = static_cast<int64_t>(start_pos);
        _pos_ids_q->load(&pos);
    } else {
        std::vector<int64_t> pos_ids_q_host(seqlen);
        for (size_t i = 0; i < seqlen; ++i) {
            pos_ids_q_host[i] = static_cast<int64_t>(start_pos + i);
        }
        _pos_ids_q->load(pos_ids_q_host.data());
    }

    // 3. Transformer layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        forward_layer(i, _x, seqlen, total_len, _pos_ids_q);
    }

    // 4. Output norm
    ensure_tensor(_x_norm, {seqlen, _meta.hs}, _meta.dtype);
    ops::rms_norm(_x_norm, _x, _weights.out_norm_w, _meta.epsilon);

    // 5. Output projection (logits)，_logits: [seqlen, voc]
    ensure_tensor(_logits, {seqlen, _meta.voc}, _meta.dtype);
    // out_embed 应该是 [voc, hs]，linear 计算 Y = X W^T，所以 Y = [seqlen, voc]
    ops::linear(_logits, _x_norm, _weights.out_embed, nullptr);

    return _logits;
}

// 跑一次前向，只返回下一个token id
int64_t Model::infer(
    int64_t *token_ids,
    size_t ntoken,
    int top_k,
    float top_p,
    float temperature) {
    // 设置设备上下文
    wginfer::core::context().setDevice(_device_type, _device_id);

    // 创建输入张量
    ensure_tensor(_input_ids_buf, {ntoken}, WGINFER_DTYPE_I64);
    _input_ids_buf->load(token_ids);

    // 确定序列长度
    size_t seqlen = ntoken;
    size_t total_len = _cache_len + seqlen;

    // 前向传播
    tensor_t logits = forward(_input_ids_buf, seqlen, total_len);

    // 本轮 forward 已把每层 K/V 写入 cache 的 [_cache_len, total_len) 区间
    _cache_len = total_len;

    // 获取最后一个 token 的 logits
    tensor_t last_logits = logits->slice(0, seqlen - 1, seqlen);
    last_logits = last_logits->view({_meta.voc});

    const bool greedy = (top_k == 1) && (top_p >= 1.0f) && (std::abs(temperature - 1.0f) < 1e-6f);
    if (greedy) {
        // Fast path: keep current argmax operator pipeline.
        ensure_tensor(_max_idx, {1}, WGINFER_DTYPE_I64);
        ensure_tensor(_max_val, {1}, _meta.dtype);
        ops::argmax(_max_idx, _max_val, last_logits);

        int64_t host_result = 0;
        wginfer::core::context().runtime().api()->memcpy_sync(
            &host_result, _max_idx->data(), sizeof(int64_t), WGINFER_MEMCPY_D2H);
        return host_result;
    }

    // Sampling path: read last-step logits to host and apply top-k/top-p/temperature.
    const WginferRuntimeAPI *api = wginfer::core::context().runtime().api();
    std::vector<float> host_logits = logits_to_host_f32(last_logits, api);
    return sample_from_logits(host_logits, top_k, top_p, temperature);
}

} // namespace wginfer::models::qwen2
