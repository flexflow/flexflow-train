#include "op-attrs/ops/reverse.h"
#include "utils/exception.h"

namespace FlexFlow {

TensorShape get_output_shape(ReverseAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(ReverseAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
