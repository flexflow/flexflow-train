#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_TRANSFORM_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_TRANSFORM_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/vector_transform.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"

namespace FlexFlow {

template <typename T, typename F, typename Out = std::invoke_result_t<F, T>>
FFOrdered<Out> transform(FFOrdered<T> const &d, F &&f) {
  return ff_ordered_of(vector_transform(vector_of(d), f));
}

} // namespace FlexFlow

#endif
