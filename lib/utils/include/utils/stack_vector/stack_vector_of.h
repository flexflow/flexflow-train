#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_STACK_VECTOR_STACK_VECTOR_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_STACK_VECTOR_STACK_VECTOR_OF_H

#include "stack_vector.h"

namespace FlexFlow {

template <size_t max_size, typename C, typename E = typename C::value_type>
stack_vector<E, max_size> stack_vector_of(C const &c) {
  stack_vector<E, max_size> result(c.cbegin(), c.cend());
  return result;
}

} // namespace FlexFlow

#endif
