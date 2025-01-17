#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_WITH_H

#include <vector>

namespace FlexFlow {

template <typename T1, typename T2, typename F, typename Result = std::invoke_result_t<F, T1, T2>>
std::vector<Result> zip_with(std::vector<T1> const &l, std::vector<T2> const &r, F &&f) {
  std::vector<Result> result;
  for (int i = 0; i < l.size() && i < r.size(); i++) {
    result.push_back(f(l.at(i), r.at(i)));
  }

  return result;
}

} // namespace FlexFlow

#endif
