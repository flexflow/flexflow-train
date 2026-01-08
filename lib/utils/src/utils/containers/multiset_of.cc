#include "utils/containers/multiset_of.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template std::multiset<T> multiset_of(std::vector<T> const &);
template std::multiset<T> multiset_of(std::set<T> const &);

} // namespace FlexFlow
