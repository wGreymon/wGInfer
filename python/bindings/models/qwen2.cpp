#include "bindings.hpp"
#include "../core/core.hpp"

#include <pybind11/stl.h>

#include "models/qwen2/model.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace qwen2 = wginfer::models::qwen2;

namespace wginfer::pybind {

namespace {

void assign_global_weight(
    qwen2::ModelWeights &weights,
    const std::string &name,
    const wginfer::tensor_t &tensor) {
    if (name == "in_embed") {
        weights.in_embed = tensor;
        return;
    }
    if (name == "out_embed") {
        weights.out_embed = tensor;
        return;
    }
    if (name == "out_norm_w") {
        weights.out_norm_w = tensor;
        return;
    }
    throw std::invalid_argument("Unknown global Qwen2 weight: " + name);
}

void assign_layer_weight(
    qwen2::ModelWeights &weights,
    const std::string &name,
    size_t layer_idx,
    const wginfer::tensor_t &tensor) {
    if (name == "attn_norm_w") {
        weights.attn_norm_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_q_w") {
        weights.attn_q_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_q_b") {
        weights.attn_q_b.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_k_w") {
        weights.attn_k_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_k_b") {
        weights.attn_k_b.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_v_w") {
        weights.attn_v_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_v_b") {
        weights.attn_v_b.at(layer_idx) = tensor;
        return;
    }
    if (name == "attn_o_w") {
        weights.attn_o_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "mlp_norm_w") {
        weights.mlp_norm_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "mlp_gate_w") {
        weights.mlp_gate_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "mlp_up_w") {
        weights.mlp_up_w.at(layer_idx) = tensor;
        return;
    }
    if (name == "mlp_down_w") {
        weights.mlp_down_w.at(layer_idx) = tensor;
        return;
    }
    throw std::invalid_argument("Unknown per-layer Qwen2 weight: " + name);
}

class PyQwen2Model {
public:
    PyQwen2Model(const qwen2::ModelMeta &meta, wginferDeviceType_t device, int device_id)
        : model_(std::make_unique<qwen2::Model>(meta, device, device_id)) {
    }

    const qwen2::ModelMeta &meta() const {
        return model_->meta();
    }

    void reset_cache() const {
        model_->reset_cache();
    }

    void set_weight(const std::string &name, const std::shared_ptr<PyTensor> &tensor) {
        assign_global_weight(model_->weights(), name, tensor->tensor());
    }

    void set_layer_weight(const std::string &name, size_t layer_idx, const std::shared_ptr<PyTensor> &tensor) {
        if (layer_idx >= model_->meta().nlayer) {
            throw std::out_of_range("Qwen2 layer_idx out of range");
        }
        assign_layer_weight(model_->weights(), name, layer_idx, tensor->tensor());
    }

    int64_t infer(
        const std::vector<int64_t> &token_ids,
        int top_k,
        float top_p,
        float temperature) const {
        std::vector<int64_t> tokens = token_ids;
        return model_->infer(
            tokens.data(),
            tokens.size(),
            top_k,
            top_p,
            temperature);
    }

private:
    std::unique_ptr<qwen2::Model> model_;
};

} // namespace

void bind_qwen2(py::module_ &m) {
    py::class_<qwen2::ModelMeta>(m, "Qwen2Meta")
        .def(py::init<>())
        .def_readwrite("dtype", &qwen2::ModelMeta::dtype)
        .def_readwrite("nlayer", &qwen2::ModelMeta::nlayer)
        .def_readwrite("hs", &qwen2::ModelMeta::hs)
        .def_readwrite("nh", &qwen2::ModelMeta::nh)
        .def_readwrite("nkvh", &qwen2::ModelMeta::nkvh)
        .def_readwrite("dh", &qwen2::ModelMeta::dh)
        .def_readwrite("di", &qwen2::ModelMeta::di)
        .def_readwrite("maxseq", &qwen2::ModelMeta::maxseq)
        .def_readwrite("voc", &qwen2::ModelMeta::voc)
        .def_readwrite("epsilon", &qwen2::ModelMeta::epsilon)
        .def_readwrite("theta", &qwen2::ModelMeta::theta)
        .def_readwrite("end_token", &qwen2::ModelMeta::end_token);

    py::class_<PyQwen2Model>(m, "Qwen2Model")
        .def(
            py::init<const qwen2::ModelMeta &, wginferDeviceType_t, int>(),
            py::arg("meta"),
            py::arg("device"),
            py::arg("device_id") = 0)
        .def("meta", &PyQwen2Model::meta, py::return_value_policy::reference_internal)
        .def("reset_cache", &PyQwen2Model::reset_cache)
        .def("set_weight", &PyQwen2Model::set_weight, py::arg("name"), py::arg("tensor"))
        .def("set_layer_weight", &PyQwen2Model::set_layer_weight, py::arg("name"), py::arg("layer_idx"), py::arg("tensor"))
        .def(
            "infer",
            &PyQwen2Model::infer,
            py::arg("token_ids"),
            py::arg("top_k") = 1,
            py::arg("top_p") = 1.0f,
            py::arg("temperature") = 1.0f);
}

} // namespace wginfer::pybind
