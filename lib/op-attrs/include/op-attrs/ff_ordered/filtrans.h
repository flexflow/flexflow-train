#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FILTRANS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_FILTRANS_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

template <typename F,
          typename In,
          typename Out = unwrap_optional_t<std::invoke_result_t<F, In>>>
FFOrdered<Out> filtrans(FFOrdered<In> const &v, F &&f) {
  return ff_ordered_of(filtrans(vector_of(v), f));
}

} // namespace FlexFlow

#endif
