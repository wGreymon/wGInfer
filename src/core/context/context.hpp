#pragma once

#include "wginfer.h"

#include "../core.hpp"

#include "../runtime/runtime.hpp"

#include <unordered_map>
#include <vector>

namespace wginfer::core {

class Context {
private:
    std::unordered_map<wginferDeviceType_t, std::vector<Runtime *>> _runtime_map;
    Runtime *_current_runtime = nullptr;
    Context();

public:
    ~Context();

    // Prevent copy
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Prevent move
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void setDevice(wginferDeviceType_t device_type, int device_id);
    Runtime &runtime();

    // thread-private
    friend Context &context();
};

} // namespace wginfer::core
