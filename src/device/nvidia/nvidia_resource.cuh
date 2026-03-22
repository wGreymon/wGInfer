#pragma once

#include "../device_resource.hpp"

namespace wginfer::device::nvidia {
class Resource : public wginfer::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};
} // namespace wginfer::device::nvidia
