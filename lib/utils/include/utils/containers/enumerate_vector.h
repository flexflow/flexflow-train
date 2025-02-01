#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_VECTOR_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include <map>
#include <vector>

namespace FlexFlow {

template <typename T>
std::map<nonnegative_int, T> enumerate_vector(std::vector<T> const &v) {
  std::map<nonnegative_int, T> result;
  for (nonnegative_int i : nonnegative_range(num_elements(v))) {
    result.insert({i, v.at(i.unwrap_nonnegative())});
  }
  return result;
}

} // namespace FlexFlow

#endif
