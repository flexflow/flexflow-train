#include "utils/variant.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template std::optional<std::variant<T1, T2, T3>>
    widen(std::variant<T1, T3> const &);
template std::optional<std::variant<T1, T3>>
    narrow(std::variant<T1, T2, T3> const &);
template std::optional<std::variant<T1, T3>> cast(std::variant<T1, T2> const &);

} // namespace FlexFlow
