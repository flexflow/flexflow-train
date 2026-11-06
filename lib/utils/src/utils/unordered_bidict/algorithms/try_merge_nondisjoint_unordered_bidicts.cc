#include "utils/unordered_bidict/algorithms/try_merge_nondisjoint_unordered_bidicts.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template std::optional<unordered_bidict<L, R>>
    try_merge_nondisjoint_unordered_bidicts(unordered_bidict<L, R> d1,
                                            unordered_bidict<L, R> const &d2);

} // namespace FlexFlow
