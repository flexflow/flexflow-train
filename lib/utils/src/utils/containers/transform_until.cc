#include "utils/containers/transform_until.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using T2 = value_type<1>;
using F = std::function<std::optional<T2>(T const &)>;

template std::vector<T2> transform_until(std::vector<T> const &, F &&);

} // namespace FlexFlow
