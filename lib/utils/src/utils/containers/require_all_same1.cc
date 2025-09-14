#include "utils/containers/require_all_same1.h"
#include "utils/archetypes/value_type.h"
#include <unordered_set>

namespace FlexFlow {

using T = value_type<0>;

template
  T require_all_same1(std::vector<T> const &);

template
  T require_all_same1(std::unordered_set<T> const &);

template
  T require_all_same1(std::unordered_multiset<T> const &);

} // namespace FlexFlow
