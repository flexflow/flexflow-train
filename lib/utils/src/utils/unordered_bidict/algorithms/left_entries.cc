#include "utils/unordered_bidict/algorithms/left_entries.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template std::unordered_set<L> left_entries(unordered_bidict<L, R> const &);

} // namespace FlexFlow
