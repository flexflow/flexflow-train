#include "utils/containers/map_keys2.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using F = std::function<K2(K const &, V const &)>;

template
  std::unordered_map<K2, V> map_keys(std::unordered_map<K, V> const &,
                                     F const &);

} // namespace FlexFlow
