#include "utils/containers/map_from_unordered.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;

template
  std::map<K, V> map_from_unordered(std::unordered_map<K, V> const &);

} // namespace FlexFlow
