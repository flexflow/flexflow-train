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
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW const &output_accessor,
                    GenericTensorAccessorR const &filter_accessor,
                    std::optional<GenericTensorAccessorR> const &bias_accessor) {
  if (stream.is_gpu()) {
    positive_int in_dim = input_accessor.shape.at(ff_dim_t{0_n});
    positive_int out_dim = output_accessor.shape.at(ff_dim_t{0_n});
    positive_int batch_size = positive_int{output_accessor.shape.num_elements() / out_dim};
  
    float const *bias_ptr = nullptr;
    if (bias_accessor.has_value()) {
      bias_ptr = bias_accessor.value().get<DataType::FLOAT>();
    }
    
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input_ptr=*/input_accessor.get<DataType::FLOAT>(),
        /*output_ptr=*/output_accessor.get<DataType::FLOAT>(),
        /*filter_ptr=*/filter_accessor.get<DataType::FLOAT>(),
        /*bias_ptr=*/bias_ptr,
        /*in_dim=*/in_dim.int_from_positive_int(),
        /*out_dim=*/out_dim.int_from_positive_int(),
        /*batch_size=*/batch_size.int_from_positive_int());
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
                       /*input_accessor=*/input_accessor,
                       /*output_accessor=*/output_accessor,
                       /*filter_accessor=*/filter_accessor,
                       /*bias_accessor=*/bias_accessor);
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
