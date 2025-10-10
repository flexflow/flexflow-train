#include "utils/containers/binary_merge_maps_with_right_dominating.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template
  std::unordered_map<K, V>
      binary_merge_maps_with_right_dominating(std::unordered_map<K, V> const &,
                                              std::unordered_map<K, V> const &);

} // namespace FlexFlow
