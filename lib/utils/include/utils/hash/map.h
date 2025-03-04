#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_MAP_H

#include "utils/hash-utils.h"
#include "utils/hash/pair.h"
#include <map>

namespace std {

template <typename K, typename V>
struct hash<std::map<K, V>> {
  size_t operator()(std::map<K, V> const &m) const {
    size_t result = 0;
    ::FlexFlow::unordered_container_hash(result, m);
    return result;
  }
};

} // namespace std

#endif
