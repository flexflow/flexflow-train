#include "utils/orthotope/orthotope_dim_indexed/drop_idxs_except.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  OrthotopeDimIndexed<T> drop_idxs_except(OrthotopeDimIndexed<T> const &, std::set<orthotope_dim_idx_t> const &);

} // namespace FlexFlow
