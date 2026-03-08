#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace wginfer::device::cpu {

namespace runtime_api {
int getDeviceCount() {
    return 1;
}

void setDevice(int) {
    // do nothing
}

void deviceSynchronize() {
    // do nothing
}

wginferStream_t createStream() {
    return (wginferStream_t)0; // null stream
}

void destroyStream(wginferStream_t stream) {
    // do nothing
}
void streamSynchronize(wginferStream_t stream) {
    // do nothing
}

void *mallocDevice(size_t size) {
    return std::malloc(size);
}

void freeDevice(void *ptr) {
    std::free(ptr);
}

void *mallocHost(size_t size) {
    return mallocDevice(size);
}

void freeHost(void *ptr) {
    freeDevice(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, wginferMemcpyKind_t kind) {
    std::memcpy(dst, src, size);
}

void memcpyAsync(void *dst, const void *src, size_t size, wginferMemcpyKind_t kind, wginferStream_t stream) {
    memcpySync(dst, src, size, kind);
}

static const WginferRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const WginferRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace wginfer::device::cpu
