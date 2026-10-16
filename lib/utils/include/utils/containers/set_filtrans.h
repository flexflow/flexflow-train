#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTRANS_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTRANS_SET_H

#include "utils/containers/get_element_type.h"
#include "utils/containers/unwrap_optional.h"
#include <set>

namespace FlexFlow {

template <typename F,
          typename C,
          typename In = get_element_type_t<C>,
          typename Out = unwrap_optional_t<std::invoke_result_t<F, In>>>
std::set<Out> set_filtrans(C const &c, F &&f) {
  std::set<Out> result;

  for (In const &i : c) {
    std::optional<Out> o = f(i);
    if (o.has_value()) {
      result.insert(o.value());
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
