#include "utils/containers/vector_from_idx_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template 
  std::optional<std::vector<T>> vector_from_idx_map(std::unordered_map<int, T> const &);

} // namespace FlexFlow
