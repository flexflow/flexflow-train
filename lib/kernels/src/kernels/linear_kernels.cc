#include "kernels/linear_kernels.h"
#include "kernels/linear_kernels_cpu.h"
#include "kernels/linear_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow::Kernels::Linear {

std::optional<LinearPerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                float *one_ptr,
                std::optional<Activation> activation,
                std::optional<RegularizerAttrs> regularizer,
                bool use_bias,
                DataType input_type,
                DataType weight_type,
                DataType output_type,
                int batch_size,
                int channel) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*one_ptr=*/one_ptr,
        /*activation=*/activation,
        /*regularizer=*/regularizer,
        /*use_bias=*/use_bias,
        /*input_type=*/input_type,
        /*weight_type=*/weight_type,
        /*output_type=*/output_type,
        /*batch_size=*/batch_size,
        /*channel=*/channel);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<LinearPerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input_ptr=*/input_ptr,
        /*output_ptr=*/output_ptr,
        /*filter_ptr=*/filter_ptr,
        /*bias_ptr=*/bias_ptr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
        /*input_ptr=*/input_ptr,
        /*output_ptr=*/output_ptr,
        /*filter_ptr=*/filter_ptr,
        /*bias_ptr=*/bias_ptr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  }
}

void backward_kernel(
    device_stream_t const &stream,
    std::optional<LinearPerDeviceState> const &per_device_state,
    float const *output_ptr,
    float *output_grad_ptr,
    float const *input_ptr,
    float *input_grad_ptr,
    float const *kernel_ptr,
    float *kernel_grad_ptr,
    float *bias_grad_ptr,
    int in_dim,
    int out_dim,
    int batch_size) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_ptr=*/output_ptr,
        /*output_grad_ptr=*/output_grad_ptr,
        /*input_ptr=*/input_ptr,
        /*input_grad_ptr=*/input_grad_ptr,
        /*kernel_ptr=*/kernel_ptr,
        /*kernel_grad_ptr=*/kernel_grad_ptr,
        /*bias_grad_ptr=*/bias_grad_ptr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*output_ptr=*/output_ptr,
        /*output_grad_ptr=*/output_grad_ptr,
        /*input_ptr=*/input_ptr,
        /*input_grad_ptr=*/input_grad_ptr,
        /*kernel_ptr=*/kernel_ptr,
        /*kernel_grad_ptr=*/kernel_grad_ptr,
        /*bias_grad_ptr=*/bias_grad_ptr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  }
}

void cleanup_kernel(DeviceType device_type,
                    std::optional<LinearPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow::Kernels::Linear
