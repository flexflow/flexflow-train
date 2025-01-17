#include "utils/containers/merge_maps.h"
#include "utils/archetypes/value_type.h"
#include "utils/hash/unordered_map.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template
  std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &,
                                      std::unordered_map<K, V> const &);

using C = std::vector<std::unordered_map<K, V>>;

template
  std::unordered_map<K, V> merge_maps(C const &);

using C2 = std::unordered_set<std::unordered_map<K, V>>;

template
  std::unordered_map<K, V> merge_maps(C2 const &);


} // namespace FlexFlow
