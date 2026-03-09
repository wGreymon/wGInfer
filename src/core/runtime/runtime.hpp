#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

namespace wginfer::core {

class Runtime {
private:
    wginferDeviceType_t _device_type;
    int _device_id;

    const WginferRuntimeAPI *_api;
    MemoryAllocator *_allocator;      

    bool _is_active;
    wginferStream_t _stream;

    Runtime(wginferDeviceType_t device_type, int device_id);
    void _activate();
    void _deactivate();

public:
    // Context need accquire Runtime information
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    wginferDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const WginferRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    wginferStream_t stream() const;
    void synchronize() const;
};

} // namespace wginfer::core
