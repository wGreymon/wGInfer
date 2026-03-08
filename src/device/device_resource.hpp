#pragma once
#include "wginfer.h"

#include "../utils.hpp"

namespace wginfer::device {
class DeviceResource {
private:
    wginferDeviceType_t _device_type;
    int _device_id;

public:
    DeviceResource(wginferDeviceType_t device_type, int device_id)
        : _device_type(device_type),
          _device_id(device_id) {
    }
    ~DeviceResource() = default;

    wginferDeviceType_t getDeviceType() const { return _device_type; }
    int getDeviceId() const { return _device_id; };
};
} // namespace wginfer::device
