#include "utils/containers/items.h"
#include "utils/archetypes/ordered_value_type.h"
#include <unordered_map>
#include <map>

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template std::set<std::pair<K, V>> items(std::unordered_map<K, V> const &);
template std::set<std::pair<K, V>> items(std::map<K, V> const &);

} // namespace FlexFlow
