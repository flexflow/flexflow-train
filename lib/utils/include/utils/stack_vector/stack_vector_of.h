#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_STACK_VECTOR_STACK_VECTOR_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_STACK_VECTOR_STACK_VECTOR_OF_H

#include "stack_vector.h"

namespace FlexFlow {

template <size_t MAX_SIZE, typename C, typename E = typename C::value_type>
stack_vector<E, MAX_SIZE> stack_vector_of(C const &c) {
  stack_vector<E, MAX_SIZE> result(c.cbegin(), c.cend());
  return result;
}

} // namespace FlexFlow

#endif
