#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_ZIP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_ZIP_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"

namespace FlexFlow {

template <typename T1, typename T2>
FFOrdered<std::pair<T1, T2>> zip(FFOrdered<T1> const &lhs,
                                 FFOrdered<T2> const &rhs) {
  return ff_ordered_of(zip(vector_of(lhs), vector_of(rhs)));
}

} // namespace FlexFlow

#endif
