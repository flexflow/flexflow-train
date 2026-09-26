#include "utils/containers/binary_merge_disjoint_sets.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template std::set<T> binary_merge_disjoint_sets(std::set<T> const &,
                                                std::set<T> const &);

} // namespace FlexFlow
