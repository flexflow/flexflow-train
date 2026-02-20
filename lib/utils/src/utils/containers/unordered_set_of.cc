#include "utils/containers/unordered_set_of.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::unordered_set<T> unordered_set_of(std::vector<T> const &);

using K = value_type<0>;
using V = value_type<1>;

template std::unordered_set<std::pair<K, V>>
    unordered_set_of(std::unordered_map<K, V> const &);

} // namespace FlexFlow
