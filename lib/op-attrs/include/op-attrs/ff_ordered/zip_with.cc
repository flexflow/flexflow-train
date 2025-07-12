#include "op-attrs/ff_ordered/zip_with.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using Result = value_type<2>;
using F = std::function<Result(T1 const &, T2 const &)>;

template FFOrdered<Result>
    zip_with(FFOrdered<T1> const &, FFOrdered<T2> const &, F &&);

} // namespace FlexFlow
