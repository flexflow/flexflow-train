#include "op-attrs/ops/input.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(InputAttrs const &attrs) {
  return attrs.tensor_shape;
}

ParallelTensorShape get_output_parallel_tensor_shape(InputAttrs const &attrs) {
  return lift_to_parallel(attrs.tensor_shape);
}

} // namespace FlexFlow
