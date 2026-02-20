#include "utils/containers/foldl1.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using F = std::function<T(T const &, T const &)>;

template T foldl1(std::vector<T> const &, F);

} // namespace FlexFlow
