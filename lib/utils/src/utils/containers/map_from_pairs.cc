#include "utils/containers/map_from_pairs.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template
  std::unordered_map<K, V>
      map_from_pairs(std::unordered_set<std::pair<K, V>> const &);

} // namespace FlexFlow
