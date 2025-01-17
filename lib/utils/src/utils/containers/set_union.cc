#include "utils/containers/set_union.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::unordered_set<T> set_union(std::unordered_set<T> const &,
                                  std::unordered_set<T> const &);

using T2 = ordered_value_type<0>;

template
  std::set<T2> set_union(std::set<T2> const &, std::set<T2> const &);

template
  std::unordered_set<T> set_union(std::vector<std::unordered_set<T>> const &);

} // namespace FlexFlow
