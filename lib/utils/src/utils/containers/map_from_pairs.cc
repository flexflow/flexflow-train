#include "utils/containers/map_from_pairs.h"
#include "utils/archetypes/ordered_value_type.h"
#include <unordered_set>
#include <set>
#include <vector>

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template std::map<K, V>
    map_from_pairs(std::set<std::pair<K, V>> const &);

template std::map<K, V>
    map_from_pairs(std::unordered_set<std::pair<K, V>> const &);

template std::map<K, V>
    map_from_pairs(std::vector<std::pair<K, V>> const &);

} // namespace FlexFlow
