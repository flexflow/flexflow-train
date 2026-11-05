#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_FILTRANS_WITH_IDX_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FF_ORDERED_FILTRANS_WITH_IDX_H

#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "utils/containers/filtrans_with_idx.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

template <
    typename In,
    typename F,
    typename Out = unwrap_optional_t<std::invoke_result_t<F, ff_dim_t, In>>>
FFOrdered<Out> ff_ordered_filtrans_with_idx(FFOrdered<In> const &v, F &&f) {
  return ff_ordered_of(filtrans_with_idx(
      vector_of(v),
      [&](nonnegative_int idx, In const &t) -> std::optional<Out> {
        return f(ff_dim_t{idx}, t);
      }));
}

} // namespace FlexFlow

#endif
