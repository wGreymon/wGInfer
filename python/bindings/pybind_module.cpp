#include <pybind11/pybind11.h>

#include "bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_wginfer, m) {
    m.doc() = "pybind11 bindings for the wginfer C++ core";

    py::module_ core_m = m.def_submodule("core", "Core bindings");
    py::module_ models_m = m.def_submodule("models", "Model bindings");

    wginfer::pybind::bind_core(core_m);
    wginfer::pybind::bind_qwen2(models_m);
    wginfer::pybind::bind_qwen3_5(models_m);
}
