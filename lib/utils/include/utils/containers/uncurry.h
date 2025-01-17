#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNCURRY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNCURRY_H

#include <functional>
#include <type_traits>

namespace FlexFlow {

template <typename T1, typename T2, typename F, typename Result = std::invoke_result_t<F, T1 const &, T2 const &>>
std::function<Result(std::pair<T1, T2> const &)> uncurry(F &&f) {
  return [f](std::pair<T1, T2> const &p) -> Result {
    return f(p.first, p.second);
  };
}

} // namespace FlexFlow

#endif
