#include "utils/containers/collapse_optionals.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::optional<T> collapse_optionals(std::optional<std::optional<T>> const &);

} // namespace FlexFlow
