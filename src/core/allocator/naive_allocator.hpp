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
} // namespace wginfer::core::allocators