#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_TWO_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_TWO_KEYS_H

#include <unordered_map>
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename K, typename V>
std::pair<V, V> require_two_keys(std::unordered_map<K, V> const &m, K const &k1, K const &k2) {
  ASSERT(k1 != k2);
  ASSERT(m.size() == 2);

  return {m.at(k1), m.at(k2)};
}

} // namespace FlexFlow

#endif
