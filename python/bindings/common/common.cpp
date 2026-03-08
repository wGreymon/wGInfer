#include "common/common.hpp"

#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;

namespace wginfer::pybind {

namespace {

void bind_enums(py::module_ &m) {
    py::enum_<wginferDeviceType_t>(m, "DeviceType")
        .value("CPU", WGINFER_DEVICE_CPU)
        .value("NVIDIA", WGINFER_DEVICE_NVIDIA)
        .value("METAX", WGINFER_DEVICE_METAX)
        .value("COUNT", WGINFER_DEVICE_TYPE_COUNT);

    py::enum_<wginferDataType_t>(m, "DataType")
        .value("INVALID", WGINFER_DTYPE_INVALID)
        .value("BYTE", WGINFER_DTYPE_BYTE)
        .value("BOOL", WGINFER_DTYPE_BOOL)
        .value("I8", WGINFER_DTYPE_I8)
        .value("I16", WGINFER_DTYPE_I16)
        .value("I32", WGINFER_DTYPE_I32)
        .value("I64", WGINFER_DTYPE_I64)
        .value("U8", WGINFER_DTYPE_U8)
        .value("U16", WGINFER_DTYPE_U16)
        .value("U32", WGINFER_DTYPE_U32)
        .value("U64", WGINFER_DTYPE_U64)
        .value("F8", WGINFER_DTYPE_F8)
        .value("F16", WGINFER_DTYPE_F16)
        .value("F32", WGINFER_DTYPE_F32)
        .value("F64", WGINFER_DTYPE_F64)
        .value("C16", WGINFER_DTYPE_C16)
        .value("C32", WGINFER_DTYPE_C32)
        .value("C64", WGINFER_DTYPE_C64)
        .value("C128", WGINFER_DTYPE_C128)
        .value("BF16", WGINFER_DTYPE_BF16);

    py::enum_<wginferMemcpyKind_t>(m, "MemcpyKind")
        .value("H2H", WGINFER_MEMCPY_H2H)
        .value("H2D", WGINFER_MEMCPY_H2D)
        .value("D2H", WGINFER_MEMCPY_D2H)
        .value("D2D", WGINFER_MEMCPY_D2D);
}

void bind_runtime(py::module_ &m) {
    py::class_<PyRuntimeAPI>(m, "RuntimeAPI")
        .def(py::init<wginferDeviceType_t>(), py::arg("device_type"))
        .def("get_device_count", &PyRuntimeAPI::get_device_count)
        .def("set_device", &PyRuntimeAPI::set_device, py::arg("device_id"))
        .def("device_synchronize", &PyRuntimeAPI::device_synchronize)
        .def("create_stream", &PyRuntimeAPI::create_stream)
        .def("destroy_stream", &PyRuntimeAPI::destroy_stream, py::arg("stream"))
        .def("stream_synchronize", &PyRuntimeAPI::stream_synchronize, py::arg("stream"))
        .def("malloc_device", &PyRuntimeAPI::malloc_device, py::arg("size"))
        .def("free_device", &PyRuntimeAPI::free_device, py::arg("ptr"))
        .def("malloc_host", &PyRuntimeAPI::malloc_host, py::arg("size"))
        .def("free_host", &PyRuntimeAPI::free_host, py::arg("ptr"))
        .def(
            "memcpy_sync",
            &PyRuntimeAPI::memcpy_sync,
            py::arg("dst"),
            py::arg("src"),
            py::arg("size"),
            py::arg("kind"))
        .def(
            "memcpy_async",
            &PyRuntimeAPI::memcpy_async,
            py::arg("dst"),
            py::arg("src"),
            py::arg("size"),
            py::arg("kind"),
            py::arg("stream"));
}

void bind_tensor(py::module_ &m) {
    py::class_<PyTensor, std::shared_ptr<PyTensor>>(m, "Tensor")
        .def(
            py::init(&PyTensor::create),
            py::arg("shape") = py::none(),
            py::arg("dtype") = WGINFER_DTYPE_F32,
            py::arg("device") = WGINFER_DEVICE_CPU,
            py::arg("device_id") = 0)
        .def("shape", &PyTensor::shape)
        .def("strides", &PyTensor::strides)
        .def("ndim", &PyTensor::ndim)
        .def("dtype", &PyTensor::dtype)
        .def("device_type", &PyTensor::device_type)
        .def("device_id", &PyTensor::device_id)
        .def("data_ptr", &PyTensor::data_ptr)
        .def("debug", &PyTensor::debug)
        .def("load", &PyTensor::load, py::arg("src"))
        .def("is_contiguous", &PyTensor::is_contiguous)
        .def("numel", &PyTensor::numel)
        .def("view", &PyTensor::view)
        .def("permute", &PyTensor::permute)
        .def("slice", &PyTensor::slice, py::arg("dim"), py::arg("start"), py::arg("end"))
        .def("__repr__", [](const PyTensor &self) {
            return "<Tensor shape=" + py::repr(self.shape()).cast<std::string>() + ">";
        });
}

void bind_ops(py::module_ &m) {
    py::class_<PyOps>(m, "Ops")
        .def_static("add", &PyOps::add, py::arg("c"), py::arg("a"), py::arg("b"))
        .def_static("argmax", &PyOps::argmax, py::arg("max_idx"), py::arg("max_val"), py::arg("vals"))
        .def_static("embedding", &PyOps::embedding, py::arg("out"), py::arg("index"), py::arg("weight"))
        .def_static(
            "linear",
            [](const std::shared_ptr<PyTensor> &out,
               const std::shared_ptr<PyTensor> &inp,
               const std::shared_ptr<PyTensor> &weight,
               py::object bias) {
                std::shared_ptr<PyTensor> bias_tensor;
                if (!bias.is_none()) {
                    bias_tensor = bias.cast<std::shared_ptr<PyTensor>>();
                }
                PyOps::linear(out, inp, weight, bias_tensor);
            },
            py::arg("out"),
            py::arg("inp"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def_static("rearrange", &PyOps::rearrange, py::arg("out"), py::arg("inp"))
        .def_static("rms_norm", &PyOps::rms_norm, py::arg("out"), py::arg("inp"), py::arg("weight"), py::arg("eps"))
        .def_static("rope", &PyOps::rope, py::arg("out"), py::arg("inp"), py::arg("pos_ids"), py::arg("theta"))
        .def_static(
            "self_attention",
            &PyOps::self_attention,
            py::arg("attn_val"),
            py::arg("q"),
            py::arg("k"),
            py::arg("v"),
            py::arg("scale"))
        .def_static("swiglu", &PyOps::swiglu, py::arg("out"), py::arg("gate"), py::arg("up"));
}

} // namespace

void bind_common(py::module_ &m) {
    bind_enums(m);
    bind_runtime(m);
    bind_tensor(m);
    bind_ops(m);
}

} // namespace wginfer::pybind
