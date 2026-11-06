#include "utils/containers/filter_values.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;
using F = std::function<bool(V const &)>;

template std::map<K, V> filter_values(std::map<K, V> const &m, F const &f);

template std::unordered_map<K, V> filter_values(std::unordered_map<K, V> const &m, F const &f);

} // namespace FlexFlow
