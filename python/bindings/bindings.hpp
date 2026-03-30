#pragma once

#include <pybind11/pybind11.h>

namespace wginfer::pybind {

void bind_core(pybind11::module_ &m);
void bind_qwen2(pybind11::module_ &m);
void bind_qwen3_5(pybind11::module_ &m);

} // namespace wginfer::pybind
