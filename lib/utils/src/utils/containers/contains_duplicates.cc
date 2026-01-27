#include "utils/containers/contains_duplicates.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template bool contains_duplicates(std::vector<T> const &);

template bool contains_duplicates(std::unordered_multiset<T> const &);

template bool contains_duplicates(std::multiset<T> const &);

} // namespace FlexFlow
