#include "utils/containers/intersection.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::unordered_set<T> intersection(std::unordered_set<T> const &, std::unordered_set<T> const &);
template 
  std::optional<std::unordered_set<T>> intersection(std::vector<std::unordered_set<T>> const &);

using T2 = ordered_value_type<0>;

template
  std::set<T2> intersection(std::set<T2> const &, std::set<T2> const &);
template 
  std::optional<std::set<T2>> intersection(std::vector<std::set<T2>> const &);

} // namespace FlexFlow
