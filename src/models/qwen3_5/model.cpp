#include "model.hpp"

#include "../../utils.hpp"

namespace wginfer::models::qwen3_5 {

namespace {

bool is_supported_layer_type(LayerType layer_type) {
    return layer_type == LayerType::LinearAttention || layer_type == LayerType::FullAttention;
}

} // namespace

Model::Model(
    const ModelMeta &meta,
    const std::vector<LayerType> &layer_types,
    wginferDeviceType_t device_type,
    int device_id)
    : _meta(meta),
      _layer_types(layer_types),
      _device_type(device_type),
      _device_id(device_id) {
    CHECK_ARGUMENT(_meta.nlayer == _layer_types.size(), "Qwen3.5 layer_types size must match nlayer");
    for (LayerType layer_type : _layer_types) {
        CHECK_ARGUMENT(is_supported_layer_type(layer_type), "Qwen3.5 layer_types contains an unsupported value");
    }
}

const ModelMeta &Model::meta() const {
    return _meta;
}

const std::vector<LayerType> &Model::layerTypes() const {
    return _layer_types;
}

wginferDeviceType_t Model::deviceType() const {
    return _device_type;
}

int Model::deviceId() const {
    return _device_id;
}

void Model::reset_cache() {
    // Placeholder for future linear/full attention cache management.
}

} // namespace wginfer::models::qwen3_5
