#include "kernels/dropout_kernels.h"
#include "kernels/dropout_kernels_cpu.h"
#include "kernels/dropout_kernels_gpu.h"

namespace FlexFlow::Kernels::Dropout {

std::optional<DropoutPerDeviceState> 
  init_kernel(DeviceType device_type,
              PerDeviceFFHandle const &handle,
                                float rate,
                                unsigned long long seed,
                                ArrayShape const &output_domain,
                                Allocator &allocator) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
                           /*handle=*/handle,
                           /*rate=*/rate,
                           /*seed=*/seed,
                           /*output_domain=*/output_domain,
                           /*allocator=*/allocator);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
  
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<DropoutPerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*per_device_state=*/per_device_state.value(),
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_forward_kernel(
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr);
  }
  
}

void backward_kernel(device_stream_t const &stream,
                     std::optional<DropoutPerDeviceState> const &per_device_state,
                     float const *output_grad_ptr,
                     float *input_grad_ptr) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*per_device_state=*/per_device_state.value(),
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_backward_kernel( 
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  }
  
}

void cleanup_kernel(DeviceType device_type,
                    Allocator &allocator,
                    std::optional<DropoutPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(allocator, per_device_state.value()); 
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
