#include "op-attrs/ops/layer_norm.h"
#include "utils/exceptions.h"

namespace FlexFlow {

// todo: maybe we need to set the degree of parallel_dim
ParallelTensorShape get_output_shape(LayerNormAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() < 2) {
    throw mk_runtime_error("LayerNorm: input must have at least 2 dimensions");
  }
  ParallelTensorShape output = input;
  // output degree is same as input degree
  return output;
}

} // namespace FlexFlow