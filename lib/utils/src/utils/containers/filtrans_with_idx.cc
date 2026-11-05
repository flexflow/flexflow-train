#include "utils/containers/filtrans_with_idx.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using In = value_type<0>;
using Out = value_type<1>;
using F = std::function<std::optional<Out>(nonnegative_int, In const &)>;

template std::vector<Out> filtrans_with_idx(std::vector<In> const &, F &&);

} // namespace FlexFlow
