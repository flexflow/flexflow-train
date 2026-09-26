#include "utils/containers/merge_disjoint_sets.h"
#include "utils/archetypes/ordered_value_type.h"
#include <vector>

namespace FlexFlow {

using T = ordered_value_type<0>;

std::set<T> merge_disjoint_sets(std::vector<std::set<T>> const &);

} // namespace FlexFlow
