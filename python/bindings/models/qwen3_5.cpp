#include "bindings.hpp"
#include "../core/core.hpp"

#include <pybind11/stl.h>

#include "models/qwen3_5/model.hpp"

#include <memory>
#include <vector>

namespace py = pybind11;
namespace qwen3_5 = wginfer::models::qwen3_5;

namespace wginfer::pybind {

namespace {

class PyQwen3_5Model {
public:
    PyQwen3_5Model(
        const qwen3_5::ModelMeta &meta,
        const std::vector<qwen3_5::LayerType> &layer_types,
        wginferDeviceType_t device,
        int device_id)
        : model_(std::make_unique<qwen3_5::Model>(meta, layer_types, device, device_id)) {
    }

    const qwen3_5::ModelMeta &meta() const {
        return model_->meta();
    }

    const std::vector<qwen3_5::LayerType> &layer_types() const {
        return model_->layerTypes();
    }

    void reset_cache() const {
        model_->reset_cache();
    }

private:
    std::unique_ptr<qwen3_5::Model> model_;
};

} // namespace

void bind_qwen3_5(py::module_ &m) {
    py::enum_<qwen3_5::LayerType>(m, "Qwen3_5LayerType")
        .value("LINEAR_ATTENTION", qwen3_5::LayerType::LinearAttention)
        .value("FULL_ATTENTION", qwen3_5::LayerType::FullAttention);

    py::class_<qwen3_5::ModelMeta>(m, "Qwen3_5Meta")
        .def(py::init<>())
        .def_readwrite("dtype", &qwen3_5::ModelMeta::dtype)
        .def_readwrite("nlayer", &qwen3_5::ModelMeta::nlayer)
        .def_readwrite("hs", &qwen3_5::ModelMeta::hs)
        .def_readwrite("nh", &qwen3_5::ModelMeta::nh)
        .def_readwrite("nkvh", &qwen3_5::ModelMeta::nkvh)
        .def_readwrite("dh", &qwen3_5::ModelMeta::dh)
        .def_readwrite("di", &qwen3_5::ModelMeta::di)
        .def_readwrite("maxseq", &qwen3_5::ModelMeta::maxseq)
        .def_readwrite("voc", &qwen3_5::ModelMeta::voc)
        .def_readwrite("epsilon", &qwen3_5::ModelMeta::epsilon)
        .def_readwrite("theta", &qwen3_5::ModelMeta::theta)
        .def_readwrite("end_token", &qwen3_5::ModelMeta::end_token)
        .def_readwrite("tie_word_embeddings", &qwen3_5::ModelMeta::tie_word_embeddings)
        .def_readwrite("full_attention_interval", &qwen3_5::ModelMeta::full_attention_interval)
        .def_readwrite("linear_num_key_heads", &qwen3_5::ModelMeta::linear_num_key_heads)
        .def_readwrite("linear_num_value_heads", &qwen3_5::ModelMeta::linear_num_value_heads)
        .def_readwrite("linear_key_head_dim", &qwen3_5::ModelMeta::linear_key_head_dim)
        .def_readwrite("linear_value_head_dim", &qwen3_5::ModelMeta::linear_value_head_dim)
        .def_readwrite("linear_conv_kernel_dim", &qwen3_5::ModelMeta::linear_conv_kernel_dim);

    py::class_<PyQwen3_5Model>(m, "Qwen3_5Model")
        .def(
            py::init<const qwen3_5::ModelMeta &, const std::vector<qwen3_5::LayerType> &, wginferDeviceType_t, int>(),
            py::arg("meta"),
            py::arg("layer_types"),
            py::arg("device"),
            py::arg("device_id") = 0)
        .def("meta", &PyQwen3_5Model::meta, py::return_value_policy::reference_internal)
        .def("layer_types", &PyQwen3_5Model::layer_types, py::return_value_policy::reference_internal)
        .def("reset_cache", &PyQwen3_5Model::reset_cache);
}

} // namespace wginfer::pybind
