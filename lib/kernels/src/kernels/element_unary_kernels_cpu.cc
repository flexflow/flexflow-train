#include "kernels/element_unary_kernels_cpu.h"

namespace FlexFlow::Kernels::ElementUnary {

void cpu_forward_kernel(ElementUnaryAttrs const &attrs,
                    PerDeviceFFHandle const &handle,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void cpu_backward_kernel(
                     ElementUnaryAttrs const &attrs,
                     PerDeviceFFHandle const &handle,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
