#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_MAP_FROM_FF_ORDERED_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_MAP_FROM_FF_ORDERED_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/nonnegative_int/num_elements.h"
#include "op-attrs/ff_dim_t.h"

namespace FlexFlow {

template <typename T>
std::unordered_map<ff_dim_t, T> map_from_ff_ordered(FFOrdered<T> const &m) {
  std::unordered_map<ff_dim_t, T> result;

  for (ff_dim_t d : ff_dim_range(num_elements(m))) {
    result.insert({d, m.at(d)});
  }

  return result;
}

} // namespace FlexFlow

#endif
