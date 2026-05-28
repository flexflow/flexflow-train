#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_VALUES_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_VALUES_STRICT_H

#include "utils/containers/generate_unordered_map.h"
#include "utils/containers/unordered_keys.h"
#include "utils/containers/require_same.h"
#include <libassert/assert.hpp>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V1, typename V2>
std::unordered_map<K, std::pair<V1, V2>>
    zip_values_strict(std::unordered_map<K, V1> const &m1,
                      std::unordered_map<K, V2> const &m2) {

  ASSERT(unordered_keys(m1) == unordered_keys(m2));

  return generate_unordered_map(require_same(unordered_keys(m1), unordered_keys(m2)), [&](K const &k) {
    return std::pair{
        m1.at(k),
        m2.at(k),
    };
  });
}

} // namespace FlexFlow

#endif
