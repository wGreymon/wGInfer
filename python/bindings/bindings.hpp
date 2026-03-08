#pragma once

#include <pybind11/pybind11.h>

namespace wginfer::pybind {

void bind_common(pybind11::module_ &m);
void bind_qwen2(pybind11::module_ &m);

} // namespace wginfer::pybind
