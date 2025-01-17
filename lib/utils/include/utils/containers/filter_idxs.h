#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTER_IDXS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTER_IDXS_H

#include <functional>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> filter_idxs(std::vector<T> const &input, std::function<bool(int)> const &f) {
  std::vector<T> result;

  for (int idx = 0; idx < input.size(); idx++) {
    if (f(idx)) {
      result.push_back(input.at(idx));
    }
  }
  
  return result;
}

} // namespace FlexFlow

#endif
