#include "utils/containers/filter_idxs.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::vector<T> filter_idxs(std::vector<T> const &, std::function<bool(int)> const &);

} // namespace FlexFlow
