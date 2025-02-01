#include "utils/containers/at_idx.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using E = value_type<0>;

template std::optional<E> at_idx(std::vector<E> const &, nonnegative_int);

} // namespace FlexFlow
