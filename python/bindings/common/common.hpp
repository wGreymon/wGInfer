#pragma once

#include <pybind11/pybind11.h>

#include "device/runtime_api.hpp"
#include "tensor/tensor.hpp"
#include "wginfer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace wginfer::pybind {

std::vector<size_t> parse_sizes(pybind11::handle obj);
pybind11::tuple to_tuple(const std::vector<size_t> &vals);
pybind11::tuple to_tuple(const std::vector<ptrdiff_t> &vals);

class PyTensor {
public:
    explicit PyTensor(wginfer::tensor_t tensor);

    static std::shared_ptr<PyTensor> create(
        pybind11::object shape,
        wginferDataType_t dtype,
        wginferDeviceType_t device,
        int device_id);

    const wginfer::tensor_t &tensor() const;
    pybind11::tuple shape() const;
    pybind11::tuple strides() const;
    size_t ndim() const;
    wginferDataType_t dtype() const;
    wginferDeviceType_t device_type() const;
    int device_id() const;
    std::uintptr_t data_ptr() const;
    void debug() const;
    void load(std::uintptr_t src_addr) const;
    bool is_contiguous() const;
    size_t numel() const;
    std::shared_ptr<PyTensor> view(pybind11::args args) const;
    std::shared_ptr<PyTensor> permute(pybind11::args args) const;
    std::shared_ptr<PyTensor> slice(size_t dim, size_t start, size_t end) const;

private:
    wginfer::tensor_t tensor_;
};

class PyRuntimeAPI {
public:
    explicit PyRuntimeAPI(wginferDeviceType_t device_type);

    int get_device_count() const;
    void set_device(int device_id) const;
    void device_synchronize() const;
    std::uintptr_t create_stream() const;
    void destroy_stream(std::uintptr_t stream) const;
    void stream_synchronize(std::uintptr_t stream) const;
    std::uintptr_t malloc_device(size_t size) const;
    void free_device(std::uintptr_t ptr) const;
    std::uintptr_t malloc_host(size_t size) const;
    void free_host(std::uintptr_t ptr) const;
    void memcpy_sync(std::uintptr_t dst, std::uintptr_t src, size_t size, wginferMemcpyKind_t kind) const;
    void memcpy_async(
        std::uintptr_t dst,
        std::uintptr_t src,
        size_t size,
        wginferMemcpyKind_t kind,
        std::uintptr_t stream) const;

private:
    const WginferRuntimeAPI *api_;
};

class PyOps {
public:
    static void add(
        const std::shared_ptr<PyTensor> &c,
        const std::shared_ptr<PyTensor> &a,
        const std::shared_ptr<PyTensor> &b);
    static void argmax(
        const std::shared_ptr<PyTensor> &max_idx,
        const std::shared_ptr<PyTensor> &max_val,
        const std::shared_ptr<PyTensor> &vals);
    static void embedding(
        const std::shared_ptr<PyTensor> &out,
        const std::shared_ptr<PyTensor> &index,
        const std::shared_ptr<PyTensor> &weight);
    static void linear(
        const std::shared_ptr<PyTensor> &out,
        const std::shared_ptr<PyTensor> &inp,
        const std::shared_ptr<PyTensor> &weight,
        const std::shared_ptr<PyTensor> &bias);
    static void rearrange(
        const std::shared_ptr<PyTensor> &out,
        const std::shared_ptr<PyTensor> &inp);
    static void rms_norm(
        const std::shared_ptr<PyTensor> &out,
        const std::shared_ptr<PyTensor> &inp,
        const std::shared_ptr<PyTensor> &weight,
        float eps);
    static void rope(
        const std::shared_ptr<PyTensor> &out,
        const std::shared_ptr<PyTensor> &inp,
        const std::shared_ptr<PyTensor> &pos_ids,
        float theta);
    static void self_attention(
        const std::shared_ptr<PyTensor> &attn_val,
        const std::shared_ptr<PyTensor> &q,
        const std::shared_ptr<PyTensor> &k,
        const std::shared_ptr<PyTensor> &v,
        float scale);
    static void swiglu(
        const std::shared_ptr<PyTensor> &out,
        const std::shared_ptr<PyTensor> &gate,
        const std::shared_ptr<PyTensor> &up);
};

} // namespace wginfer::pybind
