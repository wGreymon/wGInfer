#pragma once

#include "wginfer.h"

#include "../utils.hpp"

// c-style function pointer：equal to using get_device_count_api = int(*)()
typedef int (*get_device_count_api)();
typedef void (*set_device_api)(int);
typedef void (*device_synchronize_api)();
typedef wginferStream_t (*create_stream_api)();
typedef void (*destroy_stream_api)(wginferStream_t);
typedef void (*stream_synchronize_api)(wginferStream_t);
typedef void *(*malloc_device_api)(size_t);
typedef void (*free_device_api)(void *);
typedef void *(*malloc_host_api)(size_t);
typedef void (*free_host_api)(void *);
typedef void (*memcpy_sync_api)(void *, const void *, size_t, wginferMemcpyKind_t);
typedef void (*memcpy_async_api)(void *, const void *, size_t, wginferMemcpyKind_t, wginferStream_t);

struct WginferRuntimeAPI {
    get_device_count_api get_device_count;
    set_device_api set_device;
    device_synchronize_api device_synchronize;
    create_stream_api create_stream;
    destroy_stream_api destroy_stream;
    stream_synchronize_api stream_synchronize;
    malloc_device_api malloc_device;
    free_device_api free_device;
    malloc_host_api malloc_host;
    free_host_api free_host;
    memcpy_sync_api memcpy_sync;
    memcpy_async_api memcpy_async;
};

namespace wginfer::device {
    
const WginferRuntimeAPI *getRuntimeAPI(wginferDeviceType_t device_type);

const WginferRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const WginferRuntimeAPI *getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const WginferRuntimeAPI *getRuntimeAPI();
}
#endif

#ifdef ENABLE_METAX_API
namespace metax {
const WginferRuntimeAPI *getRuntimeAPI();
}
#endif

} // namespace wginfer::device
