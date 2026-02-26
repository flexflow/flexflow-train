#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS_AND_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS_AND_VALUES_H

#include "utils/containers/keys.h"
#include <libassert/assert.hpp>
#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V,
          typename FK,
          typename FV,
          typename K2 = std::invoke_result_t<FK, K>,
          typename V2 = std::invoke_result_t<FV, V>>
std::unordered_map<K2, V2> map_keys_and_values(
    std::unordered_map<K, V> const &m, FK const &fk, FV const &fv) {

  std::unordered_map<K2, V2> result;
  for (auto const &kv : m) {
    result.insert({fk(kv.first), fv(kv.second)});
  }

  ASSERT(keys(m).size() == keys(result).size(),
         "keys passed to map_keys must be transformed into distinct keys");

  return result;
}

} // namespace FlexFlow

#endif
