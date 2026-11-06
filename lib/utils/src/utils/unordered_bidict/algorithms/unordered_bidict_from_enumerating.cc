#include "utils/unordered_bidict/algorithms/unordered_bidict_from_enumerating.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template unordered_bidict<nonnegative_int, T>
    unordered_bidict_from_enumerating(std::vector<T> const &);

template unordered_bidict<nonnegative_int, T>
    unordered_bidict_from_enumerating(std::unordered_set<T> const &);

template unordered_bidict<nonnegative_int, T>
    unordered_bidict_from_enumerating(std::set<T> const &);

} // namespace FlexFlow
