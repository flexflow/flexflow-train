#include "utils/containers/keys.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;

template std::set<K> keys(std::unordered_map<K, V> const &);
template std::set<K> keys(std::map<K, V> const &);

} // namespace FlexFlow
