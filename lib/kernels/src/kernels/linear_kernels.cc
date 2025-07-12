#include "kernels/linear_kernels.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/linear_kernels_cpu.h"
#include "kernels/linear_kernels_gpu.h"
#include "kernels/local_cuda_allocator.h"
#include <libassert/assert.hpp>

using namespace FlexFlow::Kernels::Linear;

namespace FlexFlow {

std::optional<LinearPerDeviceState>
    linear_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
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

void linear_forward_kernel(
    device_stream_t const &stream,
    std::optional<LinearPerDeviceState> const &per_device_state,
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &input_accessor,
    GenericTensorAccessorW const &output_accessor,
    GenericTensorAccessorR const &filter_accessor,
    std::optional<GenericTensorAccessorR> const &bias_accessor) {
  if (stream.is_gpu()) {
    positive_int in_dim = dim_at_idx(input_accessor.shape.dims, ff_dim_t{1_n});
    positive_int out_dim =
        dim_at_idx(output_accessor.shape.dims, ff_dim_t{1_n});
    positive_int batch_size =
        dim_at_idx(input_accessor.shape.dims, ff_dim_t{0_n});

    float const *bias_ptr = nullptr;
    if (bias_accessor.has_value()) {
      bias_ptr = bias_accessor.value().get<DataType::FLOAT>();
    }

    ASSERT(per_device_state.has_value());
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input_ptr=*/input_accessor.get_float_ptr(),
        /*output_ptr=*/output_accessor.get_float_ptr(),
        /*filter_ptr=*/filter_accessor.get_float_ptr(),
        /*bias_ptr=*/bias_ptr,
        /*in_dim=*/in_dim.int_from_positive_int(),
        /*out_dim=*/out_dim.int_from_positive_int(),
        /*batch_size=*/batch_size.int_from_positive_int());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    linear_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input_accessor=*/input_accessor,
        /*output_accessor=*/output_accessor,
        /*filter_accessor=*/filter_accessor,
        /*bias_accessor=*/bias_accessor);
  }
}

void linear_backward_kernel(
    device_stream_t const &stream,
    std::optional<LinearPerDeviceState> const &per_device_state,
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &kernel,
    GenericTensorAccessorW const &kernel_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  if (stream.is_gpu()) {
    float *bias_grad_ptr =
        transform(bias_grad, [](GenericTensorAccessorW const &b) {
          return b.get_float_ptr();
        }).value_or(nullptr);

    positive_int in_dim = dim_at_idx(input.shape.dims, ff_dim_t{1_n});
    positive_int out_dim = dim_at_idx(output.shape.dims, ff_dim_t{1_n});
    positive_int batch_size = dim_at_idx(input.shape.dims, ff_dim_t{0_n});

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    GenericTensorAccessorW modifiable_output_grad =
        copy_tensor_accessor_r(output_grad, gpu_allocator);

    ASSERT(per_device_state.has_value());
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_ptr=*/output.get_float_ptr(),
        /*output_grad_ptr=*/modifiable_output_grad.get_float_ptr(),
        /*input_ptr=*/input.get_float_ptr(),
        /*input_grad_ptr=*/input_grad.get_float_ptr(),
        /*kernel_ptr=*/kernel.get_float_ptr(),
        /*kernel_grad_ptr=*/kernel_grad.get_float_ptr(),
        /*bias_grad_ptr=*/bias_grad_ptr,
        /*in_dim=*/in_dim.int_from_positive_int(),
        /*out_dim=*/out_dim.int_from_positive_int(),
        /*batch_size=*/batch_size.int_from_positive_int());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    linear_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*kernel=*/kernel,
        /*kernel_grad=*/kernel_grad,
        /*bias_grad=*/bias_grad);
  }
}

void linear_cleanup_kernel(
    DeviceType device_type,
    std::optional<LinearPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
