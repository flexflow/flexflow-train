#include "utils/orthotope/orthotope_dim_indexed/transform.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using Result = value_type<1>;
using F = std::function<Result(T)>;

template 
  OrthotopeDimIndexed<Result> transform(OrthotopeDimIndexed<T> const &, F &&);

} // namespace FlexFlow
