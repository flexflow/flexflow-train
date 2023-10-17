#include "op-attrs/ops/element_binary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &atts,
                                     ParallelTensorShape const &lhs,
                                     ParallelTensorShape const &rhs) {
  ParallelTensorShape output = lhs.num_dims() >= rhs.num_dims() ? lhs : rhs;
  for (int i = 0; i < output.num_dims(); i++) {
    if (i >= lhs.num_dims()) {
      output.at(ff_dim_t(i)) = rhs.at(ff_dim_t(i));
    } else if (i >= rhs.num_dims()) {
      output.at(ff_dim_t(i)) = lhs.at(ff_dim_t(i));
    } else if (lhs.at(ff_dim_t(i)).size == rhs.at(ff_dim_t(i)).size) {
      output.at(ff_dim_t(i)) = lhs.at(ff_dim_t(i));
    } else if (lhs.at(ff_dim_t(i)).size == 1) {
      output.at(ff_dim_t(i)) = rhs.at(ff_dim_t(i));
    } else if (rhs.at(ff_dim_t(i)).size == 1) {
      output.at(ff_dim_t(i)) = lhs.at(ff_dim_t(i));
    } else {
      assert(false && "Operands could not be broadcast together");
      exit(0);
    }
  }

  return output;
}

} // namespace FlexFlow