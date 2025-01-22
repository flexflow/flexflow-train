#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "op-attrs/ff_dim_t.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/set_of.h"

namespace FlexFlow {

template <typename T>
std::set<ff_dim_t> get_idxs(FFOrdered<T> const &d) {
  return transform(set_of(range(d.size())),
                   [](int i) { return ff_dim_t{nonnegative_int{i}}; });
}

} // namespace FlexFlow

#endif
