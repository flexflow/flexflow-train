#include "op-attrs/ops/weight.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

RecordFormatter as_dot(WeightAttrs const &attrs) {
  RecordFormatter r;

  for (nonnegative_int dim : attrs.tensor_shape.dims.ff_ordered) {
    r << fmt::to_string(dim);
  }

  r << fmt::to_string(attrs.tensor_shape.data_type);

  return r;
}

TensorShape get_output_shape(WeightAttrs const &attrs) {
  return attrs.tensor_shape;
}

ParallelTensorShape get_output_parallel_tensor_shape(WeightAttrs const &attrs) {
  return lift_to_parallel(attrs.tensor_shape);
}

} // namespace FlexFlow
