#include "utils/containers/foldr.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template T foldr(std::vector<T> const &,
                 T const &,
                 std::function<T(T const &, T const &)>);

} // namespace FlexFlow
