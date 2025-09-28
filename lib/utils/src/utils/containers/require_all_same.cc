#include "utils/containers/require_all_same.h"
#include "utils/archetypes/value_type.h"
#include <unordered_set>

namespace FlexFlow {

using T = value_type<0>;

template
  std::optional<T> require_all_same(std::vector<T> const &);

template
  std::optional<T> require_all_same(std::unordered_set<T> const &);

} // namespace FlexFlow
