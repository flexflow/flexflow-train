#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ONLY_KEY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ONLY_KEY_H

#include "utils/containers/contains_key.h"
#include <libassert/assert.hpp>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
V require_only_key(std::unordered_map<K, V> const &m, K const &k) {
  ASSERT(m.size() == 1);
  ASSERT(contains_key(m, k));

  return m.at(k);
}

} // namespace FlexFlow

#endif
