#include "common/common.hpp"

#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace wginfer::pybind {

std::vector<size_t> parse_sizes(py::handle obj) {
    std::vector<size_t> out;
    if (obj.is_none()) {
        return out;
    }
    for (const auto &item : py::reinterpret_borrow<py::iterable>(obj)) {
        out.push_back(py::cast<size_t>(item));
    }
    return out;
}

py::tuple to_tuple(const std::vector<size_t> &vals) {
    py::tuple out(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        out[i] = py::int_(vals[i]);
    }
    return out;
}

py::tuple to_tuple(const std::vector<ptrdiff_t> &vals) {
    py::tuple out(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        out[i] = py::int_(vals[i]);
    }
    return out;
}

PyTensor::PyTensor(wginfer::tensor_t tensor) : tensor_(std::move(tensor)) {
    if (!tensor_) {
        throw std::runtime_error("Failed to create tensor");
    }
}

std::shared_ptr<PyTensor> PyTensor::create(
    py::object shape,
    wginferDataType_t dtype,
    wginferDeviceType_t device,
    int device_id) {
    return std::make_shared<PyTensor>(
        wginfer::Tensor::create(parse_sizes(shape), dtype, device, device_id));
}

const wginfer::tensor_t &PyTensor::tensor() const {
    return tensor_;
}

py::tuple PyTensor::shape() const {
    return to_tuple(tensor_->shape());
}

py::tuple PyTensor::strides() const {
    return to_tuple(tensor_->strides());
}

size_t PyTensor::ndim() const {
    return tensor_->ndim();
}

wginferDataType_t PyTensor::dtype() const {
    return tensor_->dtype();
}

wginferDeviceType_t PyTensor::device_type() const {
    return tensor_->deviceType();
}

int PyTensor::device_id() const {
    return tensor_->deviceId();
}

std::uintptr_t PyTensor::data_ptr() const {
    return reinterpret_cast<std::uintptr_t>(tensor_->data());
}

void PyTensor::debug() const {
    tensor_->debug();
}

void PyTensor::load(std::uintptr_t src_addr) const {
    tensor_->load(reinterpret_cast<const void *>(src_addr));
}

bool PyTensor::is_contiguous() const {
    return tensor_->isContiguous();
}

size_t PyTensor::numel() const {
    return tensor_->numel();
}

std::shared_ptr<PyTensor> PyTensor::view(py::args args) const {
    std::vector<size_t> shape_vec;
    shape_vec.reserve(args.size());
    for (const auto &item : args) {
        shape_vec.push_back(py::cast<size_t>(item));
    }
    return std::make_shared<PyTensor>(tensor_->view(shape_vec));
}

std::shared_ptr<PyTensor> PyTensor::permute(py::args args) const {
    std::vector<size_t> order;
    order.reserve(args.size());
    for (const auto &item : args) {
        order.push_back(py::cast<size_t>(item));
    }
    return std::make_shared<PyTensor>(tensor_->permute(order));
}

std::shared_ptr<PyTensor> PyTensor::slice(size_t dim, size_t start, size_t end) const {
    return std::make_shared<PyTensor>(tensor_->slice(dim, start, end));
}

} // namespace wginfer::pybind
