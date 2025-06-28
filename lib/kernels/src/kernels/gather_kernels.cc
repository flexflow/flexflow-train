#include "kernels/gather_kernels.h"
#include "kernels/gather_kernels_cpu.h"
#include "kernels/gather_kernels_gpu.h"

namespace FlexFlow::Kernels::Gather {

std::optional<GatherPerDeviceState> init_kernel(DeviceType device_type,
                                                PerDeviceFFHandle const &handle,
                                                legion_dim_t legion_dim) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
        /*handle=*/handle,
        /*legion_dim=*/legion_dim);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<GatherPerDeviceState> const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input=*/input,
        /*index=*/index,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_forward_kernel(
        /*input=*/input,
        /*index=*/index,
        /*output=*/output);
  }
}

void backward_kernel(
    device_stream_t const &stream,
    std::optional<GatherPerDeviceState> const &per_device_state,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &index,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_grad=*/output_grad,
        /*index=*/index,
        /*input_grad=*/input_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_backward_kernel(
        /*output_grad=*/output_grad,
        /*index=*/index,
        /*input_grad=*/input_grad);
  }
}

} // namespace FlexFlow::Kernels::Gather
