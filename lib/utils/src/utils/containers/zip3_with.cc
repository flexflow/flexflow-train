#include "utils/containers/zip3_with.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using A = value_type<0>;
using B = value_type<1>;
using C = value_type<2>;
using Result = value_type<3>;
using F = std::function<Result(A, B, C)>;

template std::vector<Result> zip3_with(std::vector<A> const &,
                                       std::vector<B> const &,
                                       std::vector<C> const &,
                                       F &&);

} // namespace FlexFlow
