#include "utils/containers/set_filtrans.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include <vector>

namespace FlexFlow {

using In = value_type<0>;
using C = std::vector<In>;
using Out = ordered_value_type<1>;
using F = std::function<std::optional<Out>(In const &)>;

template std::set<Out> set_filtrans(C const &, F &&);

} // namespace FlexFlow
