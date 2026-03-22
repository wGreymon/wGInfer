#include "tensor.hpp"

#include "../utils.hpp"

#include <cstddef>
#include <cstring>
#include <iterator>
#include <numeric>
#include <sstream>
#include <vector>

#include <iostream>

namespace wginfer {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        wginferDataType_t dtype,
                        wginferDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    // 针对cpu的性能优化：
    // 如果需要创建CPU上的tensor，但是当前的runtime是GPU，不需要切换到CPU_runtime，
    // 因为GPU_API也提供hostMalloc
    if (device_type == WGINFER_DEVICE_CPU && core::context().runtime().deviceType() != WGINFER_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

wginferDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

wginferDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, wginferDataType_t dtype) {
    switch (dtype) {
    case WGINFER_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case WGINFER_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case WGINFER_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case WGINFER_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case WGINFER_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case WGINFER_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == WGINFER_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            WGINFER_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

// 连续：指元素在内存中排布方式与tensor按行优先展开的顺序一致
// 判断公式：stride[i] = stride[i+1] * shape[i+1]
bool Tensor::isContiguous() const {
    const auto& tensor_shape = shape();
    const auto& tensor_strides = strides();
    const size_t& tensor_ndim = ndim();

    // 标量总是连续的
    if (tensor_ndim == 0) {
        return true;
    }

    // size_t dtype_size = elementSize(); ×
    // pytorch中以元素数量为单位，而不是字节
    // 一维张量的步长必须为1
    if (tensor_ndim == 1) {
        return tensor_strides[0] == 1;
    }
    ptrdiff_t expected_stride = 1;

    // 从后往前检查(逐步升维)
    for (ptrdiff_t i = static_cast<ptrdiff_t>(tensor_ndim) - 1; i >= 0; i--) {
        if (tensor_strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= tensor_shape[i];
    }
    return true;
}

// 重排序列维度：不复制数据，需要调整shape和strides
// 即重排列shape，按照相同的顺序重排strides，storage和offset原样复用
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "order size != tensor ndim");

    // 检查每个维度是否只出现一次
    std::vector<bool> used(ndim(), false);
    for (auto index:order) {
        CHECK_ARGUMENT(index < ndim(), "order index out of dim range");
        CHECK_ARGUMENT(!used[index], "index repition");
        used[index] = true;
    }

    // 1. 创建新的meta
    wginfer::TensorMeta new_meta = _meta;
    for (size_t i = 0; i < order.size(); ++i) {
        new_meta.shape[i] = _meta.shape[order[i]];
        new_meta.strides[i] = _meta.strides[order[i]];
    }

    // 不需要复制为新的数据，所以storage不用改变
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

// view:改变张量的形状，不复制数据
// offset不变，根据新的shape计算新的strides
// 连续型数据张量：直接重塑meta即可
// TODO: 非连续情况
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 检查元素总数
    size_t new_numel = 1;
    for (auto num : shape) {
        new_numel *= num;
    }
    CHECK_ARGUMENT(new_numel == numel(), "view size match");

    // 如果张量是连续的，直接重塑即可
    if (isContiguous()) {
        TensorMeta new_meta = _meta;
        new_meta.shape = shape;

        // 计算新的 strides（从后往前）
        new_meta.strides.resize(shape.size());
        ptrdiff_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
            new_meta.strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(shape[i]);
        }

        return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
    }

    // 非连续张量暂时不支持
    return nullptr;
}

// 切片：返回原数组的某个部分
// 不复制数据只调整shape和offset，在底层和原本张量共享数据
// stride不变，因为底层内存的位置并没有改动
// 张量在内存中布局的关键：offset(起始位置)、shape(每个维度的范围)、strides(如何遍历：遍历到不同维度的步长)
// offset是字节单位，strides是元素单位(个数)
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 1. 边界检查
    CHECK_ARGUMENT(dim < ndim(), "dim out of range");
    CHECK_ARGUMENT(start < end, "start must less than end");
    CHECK_ARGUMENT(end <= shape()[dim], "end out of range");

    // 2. 创建新的meta
    wginfer::TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;

    // 3. 计算offset
    // strides以元素为单位，计算每个维度上元素的偏移量；offset以字节为单位，记录该张量在storage中的起始位置
    size_t new_offset = _offset + start * strides()[dim] * elementSize();

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr, "src must not be null");
    CHECK_ARGUMENT(isContiguous(), "load only supports contiguous tensors");

    core::context().setDevice(this->deviceType(), this->deviceId());
    const WginferRuntimeAPI *api = core::context().runtime().api();

    size_t size_bytes = this->numel() * this->elementSize();
    wginferMemcpyKind_t kind =
        (this->deviceType() == WGINFER_DEVICE_CPU) ? WGINFER_MEMCPY_H2H : WGINFER_MEMCPY_H2D;

    api->memcpy_sync(this->data(), src_, size_bytes, kind);
}


tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(wginferDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace wginfer