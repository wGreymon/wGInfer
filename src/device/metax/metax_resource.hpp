#pragma once

#include "../device_resource.hpp"

namespace wginfer::device::metax {
class Resource : public wginfer::device::DeviceResource {
public:
    explicit Resource(int device_id);
    ~Resource() = default;
};
} // namespace wginfer::device::metax
