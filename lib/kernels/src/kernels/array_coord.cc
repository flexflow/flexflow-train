#include "kernels/array_coord.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/ff_ordered/get_idxs.h"
#include <vector>

namespace FlexFlow {

ArrayCoord array_coord_drop_dims(
    ArrayCoord const &coord,
    std::function<bool(ff_dim_t)> const &should_drop_dim) {
  std::vector<nonnegative_int> result;
  for (ff_dim_t idx : get_idxs(coord.ff_ordered)) {
    if (!should_drop_dim(idx)) {
      result.push_back(coord.ff_ordered.at(idx));
    }
  }

  return ArrayCoord{ff_ordered_of(result)};
}

} // namespace FlexFlow
