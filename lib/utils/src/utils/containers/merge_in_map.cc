#include "utils/containers/merge_in_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template
  void merge_in_map(std::unordered_map<K, V> const &,
                    std::unordered_map<K, V> &);

} // namespace FlexFlow
