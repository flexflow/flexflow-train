#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "op-attrs/ff_dim_t.h"
#include "utils/containers/count.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

template <typename T>
std::vector<ff_dim_t> get_idxs(FFOrdered<T> const &d) {
  return transform(count(d.size()),
                   [](int i) { return ff_dim_t{nonnegative_int{i}}; });
}

} // namespace FlexFlow

#endif
