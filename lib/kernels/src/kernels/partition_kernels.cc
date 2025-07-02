#include "kernels/partition_kernels.h"
#include "kernels/partition_kernels_cpu.h"
#include "kernels/partition_kernels_gpu.h"

namespace FlexFlow::Kernels::Repartition {

std::optional<RepartitionPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                DataType data_type) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*data_type=*/data_type);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void forward_kernel(
    device_stream_t const &stream,
    std::optional<RepartitionPerDeviceState> const &per_device_state,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input=*/input,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_forward_kernel(
        /*input=*/input,
        /*output=*/output);
  }
}

void backward_kernel(
    device_stream_t const &stream,
    std::optional<RepartitionPerDeviceState> const &per_device_state,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_forward_kernel(
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);
  }
}

void cleanup_kernel(
    DeviceType device_type,
    std::optional<RepartitionPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow::Kernels::Repartition
