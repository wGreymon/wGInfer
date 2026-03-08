#include "common/common.hpp"

namespace wginfer::pybind {

PyRuntimeAPI::PyRuntimeAPI(wginferDeviceType_t device_type)
    : api_(wginfer::device::getRuntimeAPI(device_type)) {
}

int PyRuntimeAPI::get_device_count() const {
    return api_->get_device_count();
}

void PyRuntimeAPI::set_device(int device_id) const {
    api_->set_device(device_id);
}

void PyRuntimeAPI::device_synchronize() const {
    api_->device_synchronize();
}

std::uintptr_t PyRuntimeAPI::create_stream() const {
    return reinterpret_cast<std::uintptr_t>(api_->create_stream());
}

void PyRuntimeAPI::destroy_stream(std::uintptr_t stream) const {
    api_->destroy_stream(reinterpret_cast<wginferStream_t>(stream));
}

void PyRuntimeAPI::stream_synchronize(std::uintptr_t stream) const {
    api_->stream_synchronize(reinterpret_cast<wginferStream_t>(stream));
}

std::uintptr_t PyRuntimeAPI::malloc_device(size_t size) const {
    return reinterpret_cast<std::uintptr_t>(api_->malloc_device(size));
}

void PyRuntimeAPI::free_device(std::uintptr_t ptr) const {
    api_->free_device(reinterpret_cast<void *>(ptr));
}

std::uintptr_t PyRuntimeAPI::malloc_host(size_t size) const {
    return reinterpret_cast<std::uintptr_t>(api_->malloc_host(size));
}

void PyRuntimeAPI::free_host(std::uintptr_t ptr) const {
    api_->free_host(reinterpret_cast<void *>(ptr));
}

void PyRuntimeAPI::memcpy_sync(
    std::uintptr_t dst,
    std::uintptr_t src,
    size_t size,
    wginferMemcpyKind_t kind) const {
    api_->memcpy_sync(
        reinterpret_cast<void *>(dst),
        reinterpret_cast<const void *>(src),
        size,
        kind);
}

void PyRuntimeAPI::memcpy_async(
    std::uintptr_t dst,
    std::uintptr_t src,
    size_t size,
    wginferMemcpyKind_t kind,
    std::uintptr_t stream) const {
    api_->memcpy_async(
        reinterpret_cast<void *>(dst),
        reinterpret_cast<const void *>(src),
        size,
        kind,
        reinterpret_cast<wginferStream_t>(stream));
}

} // namespace wginfer::pybind
