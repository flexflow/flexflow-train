#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed_from_idx_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template 
  std::optional<OrthotopeDimIndexed<T>> orthotope_dim_indexed_from_idx_map(std::unordered_map<orthotope_dim_idx_t, T> const &);

} // namespace FlexFlow
