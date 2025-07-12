#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ALL_ARE_TRUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ALL_ARE_TRUE_H

namespace FlexFlow {

template <typename Container>
bool all_are_true(Container const &c) {
  bool result = true;
  for (bool b : c) {
    result &= b;
  }
  return result;
}

} // namespace FlexFlow

#endif
