#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_LIFT_OPTIONAL_THROUGH_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_LIFT_OPTIONAL_THROUGH_MAP_H

#include <optional>
#include <unordered_map>
#include "utils/containers/values.h"
#include <libassert/assert.hpp>
#include "utils/containers/all_of.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

template <typename K, typename V>
static std::optional<std::unordered_map<K, V>> lift_optional_through_map(std::unordered_map<K, std::optional<V>> const &m) {
  ASSERT(!m.empty());

  std::unordered_multiset<std::optional<V>> m_values = values(m);

  bool has_all_values 
    = all_of(m_values, [](std::optional<V> const &t) -> bool {
                       return t.has_value();
                     });

  bool has_no_values 
    = all_of(m_values, [](std::optional<V> const &t) -> bool {
                       return !t.has_value();
                     });

  ASSERT(has_all_values || has_no_values);
  if (has_no_values) {
    return std::nullopt;
  } else {
    return map_values(m, 
                      [](std::optional<V> const &t) -> V {
                        return t.value(); 
                      });
  }
}


} // namespace FlexFlow

#endif
