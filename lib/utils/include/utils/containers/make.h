#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAKE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAKE_H

namespace FlexFlow {

template <typename T>
decltype(auto) make() {
  return [](auto const &x) { return T{x}; };
}

} // namespace FlexFlow

#endif
