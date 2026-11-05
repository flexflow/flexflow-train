#include "kernels/reduce_kernels.h"
#include "kernels/reduce_kernels_cpu.h"
#include "kernels/reduce_kernels_gpu.h"
#include "op-attrs/ops/reduce.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::optional<ReducePerDeviceState>
    reduce_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       ReduceAttrs const &attrs,
                       TensorShape const &input_shape) {
  if (device_type == DeviceType::GPU) {
    TensorShape output_shape = reduce_get_output_shape(attrs, input_shape);

    nonnegative_int reduction_size = get_num_elements(input_shape.dims) /
                                     get_num_elements(output_shape.dims);

    return reduce_gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*reduce_op=*/attrs.op,
        /*reduction_size=*/reduction_size.size_t_from_nonnegative_int(),
        /*input_shape=*/input_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void reduce_forward_kernel(
    device_stream_t const &stream,
    std::optional<ReducePerDeviceState> const &per_device_state,
    ReduceAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    reduce_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input_ptr=*/input.get_float_ptr(),
        /*output_ptr=*/output.get_float_ptr());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    reduce_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  }
}

void reduce_backward_kernel(
    device_stream_t const &stream,
    std::optional<ReducePerDeviceState> const &per_device_state,
    ReduceAttrs const &attrs,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    reduce_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_grad_ptr=*/output_grad.get_float_ptr(),
        /*input_grad_ptr=*/input_grad.get_float_ptr());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    reduce_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);
  }
}

} // namespace FlexFlow
