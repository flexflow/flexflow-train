#include "utils/unordered_bidict/algorithms/unordered_bidict_right_entries.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template std::unordered_set<R>
    unordered_bidict_right_entries(unordered_bidict<L, R> const &);

} // namespace FlexFlow
