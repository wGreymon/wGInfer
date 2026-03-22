#include "nvidia_resource.cuh"

namespace wginfer::device::nvidia {

Resource::Resource(int device_id) : wginfer::device::DeviceResource(WGINFER_DEVICE_NVIDIA, device_id) {}

} // namespace wginfer::device::nvidia
