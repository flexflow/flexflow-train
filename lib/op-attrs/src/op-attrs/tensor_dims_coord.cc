#include "op-attrs/tensor_dims_coord.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/ff_ordered/get_idxs.h"

namespace FlexFlow {

nonnegative_int
    tensor_dims_coord_get_num_dims(TensorDimsCoord const &tensor_dims_coord) {
  return nonnegative_int{tensor_dims_coord.ff_ordered.size()};
}

TensorDimsCoord tensor_dims_coord_drop_dims(
    TensorDimsCoord const &coord,
    std::function<bool(ff_dim_t)> const &should_drop_dim) {
  std::vector<nonnegative_int> result;
  for (ff_dim_t idx : get_idxs(coord.ff_ordered)) {
    if (!should_drop_dim(idx)) {
      result.push_back(coord.ff_ordered.at(idx));
    }
  }

  return TensorDimsCoord{ff_ordered_of(result)};
}

} // namespace FlexFlow
