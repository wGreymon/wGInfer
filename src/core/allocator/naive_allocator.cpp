#include "naive_allocator.hpp"

#include "../runtime/runtime.hpp"

namespace wginfer::core::allocators {
NaiveAllocator::NaiveAllocator(const WginferRuntimeAPI *runtime_api) : MemoryAllocator(runtime_api) {
}

std::byte *NaiveAllocator::allocate(size_t size) {
    return static_cast<std::byte *>(_api->malloc_device(size));
}

void NaiveAllocator::release(std::byte *memory) {
    _api->free_device(memory);
}
} // namespace wginfer::core::allocators