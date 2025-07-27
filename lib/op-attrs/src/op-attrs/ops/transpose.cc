#include "op-attrs/ops/transpose.h"
#include "utils/exception.h"

namespace FlexFlow {

TensorShape get_output_shape(TransposeAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(TransposeAttrs const &op_attrs,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
