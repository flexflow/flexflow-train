#include "kernels/element_unary_kernels_cpu.h"
#include "kernels/map_tensor_accessors.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "utils/exception.h"

namespace FlexFlow::Kernels::ElementUnary {

void cpu_forward_kernel(ElementUnaryAttrs const &attrs,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  switch (attrs.op_type) {
    case OperatorType::RELU:
      tensor_accessor_relu_to(input, output);
      break;
    default:
      NOT_IMPLEMENTED();
  }
}

void cpu_backward_kernel(ElementUnaryAttrs const &attrs,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &input_grad) {

  switch (attrs.op_type) {
    case OperatorType::RELU:
      // relu backward: input_grad = output_grad * (output > 0)
      map_tensor_accessors2_to(
          output_grad,
          output,
          output_grad.shape.data_type,
          [](auto grad, auto out) {
            return out > static_cast<decltype(out)>(0)
                       ? grad
                       : static_cast<decltype(grad)>(0);
          },
          input_grad);
      break;
    default:
      NOT_IMPLEMENTED();
  }
}

} // namespace FlexFlow::Kernels::ElementUnary
