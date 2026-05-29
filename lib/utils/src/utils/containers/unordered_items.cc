#include "utils/containers/unordered_items.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include <map>
#include <unordered_map>

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;

template std::unordered_set<std::pair<K, V>> unordered_items(std::unordered_map<K, V> const &);
template std::unordered_set<std::pair<K, V>> unordered_items(std::map<K, V> const &);

} // namespace FlexFlow
