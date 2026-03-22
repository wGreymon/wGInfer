#pragma once

#include "../../device/runtime_api.hpp"

#include "../storage/storage.hpp"

namespace wginfer::core {
    
class MemoryAllocator {
protected:
    const WginferRuntimeAPI *_api;
    MemoryAllocator(const WginferRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void release(std::byte *memory) = 0;
};

} // namespace wginfer::core
