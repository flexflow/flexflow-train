#include "utils/containers/get_only.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template T get_only(std::vector<T> const &);
template T get_only(std::set<T> const &);
template T get_only(std::unordered_set<T> const &);

using K = value_type<1>;
using V = value_type<2>;

template std::pair<K, V> get_only(std::unordered_map<K, V> const &);

} // namespace FlexFlow
