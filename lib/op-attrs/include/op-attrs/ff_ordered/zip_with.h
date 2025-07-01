#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_ZIP_WITH_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_ZIP_WITH_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/containers/zip_with.h"

namespace FlexFlow {

template <typename T1,
          typename T2,
          typename F,
          typename Result = std::invoke_result_t<F, T1, T2>>
FFOrdered<Result> zip_with(FFOrdered<T1> const &lhs, FFOrdered<T2> const &rhs, F &&f) {
  return FFOrdered<Result>{zip_with(vector_of(lhs), vector_of(rhs), f)};
}

} // namespace FlexFlow

#endif
