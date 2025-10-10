#include "utils/containers/merge_maps_with_right_dominating.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using C = std::vector<std::unordered_map<K, V>>;

template
  std::unordered_map<K, V> merge_maps_with_right_dominating(C const &);

} // namespace FlexFlow
