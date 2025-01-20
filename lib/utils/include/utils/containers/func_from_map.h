#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FUNC_FROM_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FUNC_FROM_MAP_H

#include "utils/containers/contains.h"
#include "utils/containers/keys.h"
#include "utils/exception.h"
#include <functional>
#include <string>
#include <unordered_map>
#include "utils/fmt/unordered_map.h"

namespace FlexFlow {

template <typename K, typename V>
std::function<V(K const &)> func_from_map(std::unordered_map<K, V> const &map) {
  return [map](K const &key) -> V {
    if (!contains(keys(map), key)) {
      throw mk_runtime_error(fmt::format(
          "Key {} is not present in the underlying map {}", key, map));
    }
    return map.at(key);
  };
}

} // namespace FlexFlow

#endif
