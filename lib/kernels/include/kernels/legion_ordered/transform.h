#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LEGION_ORDERED_TRANSFORM_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LEGION_ORDERED_TRANSFORM_H

#include "kernels/legion_ordered/legion_ordered.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/vector_transform.h"

namespace FlexFlow {

template <typename T, typename F, typename Out = std::invoke_result_t<F, T>>
LegionOrdered<Out> transform(LegionOrdered<T> const &d, F &&f) {
  return LegionOrdered<Out>{vector_transform(vector_of(d), f)};
}

} // namespace FlexFlow

#endif
