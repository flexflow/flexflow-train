#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed_of.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  OrthotopeDimIndexed<T> orthotope_dim_indexed_of(std::vector<T> const &);

} // namespace FlexFlow
