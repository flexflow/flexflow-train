#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_VALUES_STRICT_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_VALUES_STRICT_WITH_H

#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/require_same.h"
#include <libassert/assert.hpp>
#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V1,
          typename V2,
          typename F,
          typename Out = std::invoke_result_t<F, V1, V2>>
std::unordered_map<K, Out>
    zip_values_strict_with(std::unordered_map<K, V1> const &m1,
                           std::unordered_map<K, V2> const &m2,
                           F &&f) {

  ASSERT(keys(m1) == keys(m2));

  return generate_map(require_same(keys(m1), keys(m2)),
                      [&](K const &k) -> Out { return f(m1.at(k), m2.at(k)); });
}

} // namespace FlexFlow

#endif
