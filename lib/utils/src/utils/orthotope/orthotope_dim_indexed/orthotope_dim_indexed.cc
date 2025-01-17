#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  struct OrthotopeDimIndexed<T>;

using T2 = ordered_value_type<0>;

// template 
//   bool operator<(OrthotopeDimIndexed<T2> const &, OrthotopeDimIndexed<T2> const &);

} // namespace FlexFlow
