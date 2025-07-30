#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_ORDERING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_ORDERING_H

#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"

namespace FlexFlow {

template <typename T>
DimOrdering<T> make_default_dim_ordering() {
  return DimOrdering<T>{
    [](T const &lhs, T const &rhs) -> bool { return lhs < rhs; },
  };
}

template <typename T>
DimOrdering<T> make_reversed_dim_ordering() {
  return DimOrdering<T>{
    [](T const &lhs, T const &rhs) -> bool { return rhs < lhs; },
  };
}

template <typename T>
DimOrdering<T> make_dim_ordering_from_vector(std::vector<T> const &v) {
  bidict<nonnegative_int, T> v_as_map = bidict_from_enumerating(v);

  return DimOrdering<T>{
    [=](T const &lhs, T const &rhs) -> bool {
      return v_as_map.at_r(lhs) <= v_as_map.at_r(rhs);
    },
  };
}

} // namespace FlexFlow

#endif
