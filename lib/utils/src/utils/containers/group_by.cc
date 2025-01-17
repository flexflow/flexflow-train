#include "utils/containers/group_by.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using F = std::function<K(V)>;

template 
  std::unordered_map<K, std::unordered_set<V>>
      group_by(std::unordered_set<V> const &, F &&);

template 
  std::unordered_map<K, std::vector<V>> 
      group_by(std::vector<V> const &, F &&);

using V2 = ordered_value_type<1>;
using F2 = std::function<K(V2)>;

template 
  std::unordered_map<K, std::set<V2>> 
      group_by(std::set<V2> const &, F2 &&);

} // namespace FlexFlow
