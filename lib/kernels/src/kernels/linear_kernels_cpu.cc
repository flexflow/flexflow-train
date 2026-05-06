#include "kernels/linear_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/map_tensor_accessors.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

void linear_cpu_forward_kernel(
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &projection,
    std::optional<GenericTensorAccessorR> const &bias) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  tensor_accessor_matmul_to(
      input, tensor_accessor_transpose(projection, cpu_allocator), output);

  ASSERT(attrs.use_bias == bias.has_value());
  if (bias.has_value()) {
    GenericTensorAccessorW broadcasted_bias = tensor_accessor_broadcast(
        bias.value(), output.shape.dims, cpu_allocator);
    tensor_accessor_elementwise_add_to(
        read_only_accessor_from_write_accessor(output),
        read_only_accessor_from_write_accessor(broadcasted_bias),
        output);
  }

  if (attrs.activation.has_value()) {
    switch (attrs.activation.value()) {
      case Activation::RELU:
        tensor_accessor_relu_to(read_only_accessor_from_write_accessor(output),
                                output);
        break;
      default:
        PANIC("Unhandled activation function", attrs.activation.value());
    }
  }
}

void linear_cpu_backward_kernel(
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &projection,
    GenericTensorAccessorW const &projection_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  std::optional<GenericTensorAccessorR> processed_output_grad = std::nullopt;
  if (attrs.activation.has_value()) {
    switch (attrs.activation.value()) {
      case Activation::RELU: {
        // relu backward: output_grad * (output > 0)
        // output here is POST-activation (relu output)
        // output > 0 iff pre-activation > 0 since relu(x) > 0 iff x > 0
        GenericTensorAccessorW grad_buf =
            cpu_allocator.allocate_tensor(output_grad.shape);
        map_tensor_accessors2_to(
            output_grad,
            output,
            output_grad.shape.data_type,
            [](auto grad, auto out) {
              return out > static_cast<decltype(out)>(0)
                         ? grad
                         : static_cast<decltype(grad)>(0);
            },
            grad_buf);
        processed_output_grad =
            read_only_accessor_from_write_accessor(grad_buf);
        break;
      }
      default:
        PANIC("Unhandled activation function", attrs.activation.value());
    }
  } else {
    processed_output_grad = output_grad;
  }

  tensor_accessor_matmul_to(
      processed_output_grad.value(), projection, input_grad);
  tensor_accessor_transpose_to(
      tensor_accessor_matmul(
          read_only_accessor_from_write_accessor(
              tensor_accessor_transpose(input, cpu_allocator)),
          processed_output_grad.value(),
          cpu_allocator),
      projection_grad);

  if (bias_grad.has_value()) {
    tensor_accessor_reduce_to(
        processed_output_grad.value(), ff_dim_t{0_n}, bias_grad.value());
  }
}

} // namespace FlexFlow
