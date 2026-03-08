#include <pybind11/pybind11.h>

#include "bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_wginfer, m) {
    m.doc() = "pybind11 bindings for the wginfer C++ core";

    wginfer::pybind::bind_common(m);
    wginfer::pybind::bind_qwen2(m);
}
