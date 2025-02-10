#include "utils/containers/zip_with.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using Result = value_type<2>;
using F = std::function<Result(T1, T2)>;

template std::vector<Result>
    zip_with(std::vector<T1> const &, std::vector<T2> const &, F &&);

} // namespace FlexFlow
