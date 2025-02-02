#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template bidict<nonnegative_int, T>
    bidict_from_enumerating(std::unordered_set<T> const &);

template bidict<nonnegative_int, T>
    bidict_from_enumerating(std::set<T> const &);

} // namespace FlexFlow
