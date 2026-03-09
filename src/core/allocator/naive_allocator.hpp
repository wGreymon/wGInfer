#pragma once

#include "allocator.hpp"

namespace wginfer::core::allocators {

class NaiveAllocator : public MemoryAllocator {
public:
    NaiveAllocator(const WginferRuntimeAPI *runtime_api);
    ~NaiveAllocator() = default;
    std::byte *allocate(size_t size) override;
    void release(std::byte *memory) override;
};

// TODO：内存池、缓存分配器、size class复用，避免频繁的cudaMalloc/cudaFree

} // namespace wginfer::core::allocators