#include "utils/orthotope/orthotope_dim_indexed/zip_with.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using Result = value_type<2>;
using F = std::function<Result(T1, T2)>;

template
  OrthotopeDimIndexed<Result> zip_with(OrthotopeDimIndexed<T1> const &, OrthotopeDimIndexed<T2> const &, F &&);

} // namespace FlexFlow
