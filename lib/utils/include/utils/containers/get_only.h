#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONLY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONLY_H

#include "utils/containers/maybe_get_only.h"
#include <libassert/assert.hpp>
#include "utils/optional.h"

namespace FlexFlow {

template <typename C>
typename C::value_type get_only(C const &c) {
  ASSERT(c.size() == 1);
  return maybe_get_only(c).value();
}

template <typename K, typename V>
std::pair<K, V> get_only(std::unordered_map<K, V> const &m) {
  ASSERT(m.size() == 1);

  return *m.cbegin();
}

} // namespace FlexFlow

#endif
