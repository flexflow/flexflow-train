#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NUM_ELEMENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NUM_ELEMENTS_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename T>
nonnegative_int num_elements(T const &t) {
  size_t t_size = t.size();
  if (t_size >= static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw mk_runtime_error("Cannot represent num elements as nonnegative_int");
  }
  
  return nonnegative_int{static_cast<int>(t_size)};
}

} // namespace FlexFlow

#endif
