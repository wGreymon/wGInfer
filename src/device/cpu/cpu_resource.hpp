#pragma once

#include "../device_resource.hpp"

namespace wginfer::device::cpu {
class Resource : public wginfer::device::DeviceResource {
public:
    Resource();
    ~Resource() = default;
};
} // namespace wginfer::device::cpu