#include "utils/bidict/algorithms/filter_keys.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using F = std::function<bool(K)>;

template
  bidict<K, V> filter_keys(bidict<K, V> const &, F &&);

} // namespace FlexFlow
