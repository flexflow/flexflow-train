#include "op-attrs/tensor_num_dims.h"
#include "op-attrs/ff_dim_t.h"

namespace FlexFlow {

std::set<ff_dim_t> ff_dim_idxs_for_num_dims(TensorNumDims const &num_dims) {
  return ff_dim_range(num_dims.value);
}

} // namespace FlexFlow
