#include "common/common.hpp"

#include "ops/add/op.hpp"
#include "ops/argmax/op.hpp"
#include "ops/embedding/op.hpp"
#include "ops/linear/op.hpp"
#include "ops/rearrange/op.hpp"
#include "ops/rms_norm/op.hpp"
#include "ops/rope/op.hpp"
#include "ops/self_attention/op.hpp"
#include "ops/swiglu/op.hpp"

namespace wginfer::pybind {

void PyOps::add(
    const std::shared_ptr<PyTensor> &c,
    const std::shared_ptr<PyTensor> &a,
    const std::shared_ptr<PyTensor> &b) {
    wginfer::ops::add(c->tensor(), a->tensor(), b->tensor());
}

void PyOps::argmax(
    const std::shared_ptr<PyTensor> &max_idx,
    const std::shared_ptr<PyTensor> &max_val,
    const std::shared_ptr<PyTensor> &vals) {
    wginfer::ops::argmax(max_idx->tensor(), max_val->tensor(), vals->tensor());
}

void PyOps::embedding(
    const std::shared_ptr<PyTensor> &out,
    const std::shared_ptr<PyTensor> &index,
    const std::shared_ptr<PyTensor> &weight) {
    wginfer::ops::embedding(out->tensor(), index->tensor(), weight->tensor());
}

void PyOps::linear(
    const std::shared_ptr<PyTensor> &out,
    const std::shared_ptr<PyTensor> &inp,
    const std::shared_ptr<PyTensor> &weight,
    const std::shared_ptr<PyTensor> &bias) {
    wginfer::ops::linear(
        out->tensor(),
        inp->tensor(),
        weight->tensor(),
        bias ? bias->tensor() : nullptr);
}

void PyOps::rearrange(
    const std::shared_ptr<PyTensor> &out,
    const std::shared_ptr<PyTensor> &inp) {
    wginfer::ops::rearrange(out->tensor(), inp->tensor());
}

void PyOps::rms_norm(
    const std::shared_ptr<PyTensor> &out,
    const std::shared_ptr<PyTensor> &inp,
    const std::shared_ptr<PyTensor> &weight,
    float eps) {
    wginfer::ops::rms_norm(out->tensor(), inp->tensor(), weight->tensor(), eps);
}

void PyOps::rope(
    const std::shared_ptr<PyTensor> &out,
    const std::shared_ptr<PyTensor> &inp,
    const std::shared_ptr<PyTensor> &pos_ids,
    float theta) {
    wginfer::ops::rope(out->tensor(), inp->tensor(), pos_ids->tensor(), theta);
}

void PyOps::self_attention(
    const std::shared_ptr<PyTensor> &attn_val,
    const std::shared_ptr<PyTensor> &q,
    const std::shared_ptr<PyTensor> &k,
    const std::shared_ptr<PyTensor> &v,
    float scale) {
    wginfer::ops::self_attention(attn_val->tensor(), q->tensor(), k->tensor(), v->tensor(), scale);
}

void PyOps::swiglu(
    const std::shared_ptr<PyTensor> &out,
    const std::shared_ptr<PyTensor> &gate,
    const std::shared_ptr<PyTensor> &up) {
    wginfer::ops::swiglu(out->tensor(), gate->tensor(), up->tensor());
}

} // namespace wginfer::pybind
