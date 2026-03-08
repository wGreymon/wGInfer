#include "cpu_resource.hpp"

namespace wginfer::device::cpu {
Resource::Resource() : wginfer::device::DeviceResource(WGINFER_DEVICE_CPU, 0) {}
} // namespace wginfer::device::cpu
