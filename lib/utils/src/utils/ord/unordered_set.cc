#include "utils/ord/unordered_set.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template bool operator<(std::unordered_set<T> const &,
                        std::unordered_set<T> const &);

} // namespace FlexFlow
