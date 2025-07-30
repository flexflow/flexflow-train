#include "utils/containers/require_same.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template T const &require_same(T const &, T const &);

template T const &require_same(T const &, T const &, T const &);

template T const &require_same(T const &, T const &, T const &, T const &);

} // namespace FlexFlow
