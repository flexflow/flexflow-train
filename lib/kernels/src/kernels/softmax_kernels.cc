#include "kernels/softmax_kernels.h"
#include "kernels/softmax_kernels_cpu.h"
#include "kernels/softmax_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow::Kernels::Softmax {

std::optional<SoftmaxPerDeviceState> init_kernel(DeviceType device_type,
                                                 device_handle_t const &handle,
                                                 legion_dim_t dim,
                                                 int input_n,
                                                 int input_c,
                                                 int input_h,
                                                 int input_w) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*dim=*/dim,
        /*input_n=*/input_n,
        /*input_c=*/input_c,
        /*input_h=*/input_h,
        /*input_w=*/input_w);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void forward_kernel(
    device_stream_t const &stream,
    std::optional<SoftmaxPerDeviceState> const &per_device_state,
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
                     float const *output_grad_ptr,
                     float *input_grad_ptr,
                     size_t num_elements) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output_grad_ptr=*/output_grad_ptr,
        /*input_grad_ptr=*/input_grad_ptr,
        /*num_elements=*/num_elements);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*output_grad_ptr=*/output_grad_ptr,
        /*input_grad_ptr=*/input_grad_ptr,
        /*num_elements=*/num_elements);
  }
}

void cleanup_kernel(DeviceType device_type,
                    std::optional<SoftmaxPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow::Kernels::Softmax
