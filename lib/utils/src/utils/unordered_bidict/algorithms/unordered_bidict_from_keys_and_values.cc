#include "utils/unordered_bidict/algorithms/unordered_bidict_from_keys_and_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template unordered_bidict<L, R>
    unordered_bidict_from_keys_and_values(std::vector<L> const &,
                                          std::vector<R> const &);

} // namespace FlexFlow
