#include "utils/ord/vector.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template
  bool operator<(std::vector<T> const &, std::vector<T> const &);

} // namespace FlexFlow
