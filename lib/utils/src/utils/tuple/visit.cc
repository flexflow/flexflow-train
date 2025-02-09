#include "utils/tuple/visit.h"
#include "utils/archetypes/value_type.h"
#include <variant>

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;
using Visitor = std::function<void(std::variant<T1, T2, T3> const &)>;

template void visit_tuple(std::tuple<T1, T2, T3> const &, Visitor &&);

} // namespace FlexFlow
