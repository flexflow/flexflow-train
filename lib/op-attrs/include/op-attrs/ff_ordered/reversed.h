#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_REVERSED_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_REVERSED_H

#include "op-attrs/ff_ordered/ff_ordered.h"

namespace FlexFlow {

template <typename T>
FFOrdered<T> reversed(FFOrdered<T> const &t) {
  FFOrdered<T> result(std::crbegin(t), std::crend(t));
  return result;
}

} // namespace FlexFlow

#endif
