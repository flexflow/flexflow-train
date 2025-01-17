#include "utils/containers/is_subseteq_of.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
template
  bool is_subseteq_of(std::unordered_set<T1> const &, std::unordered_set<T1> const &);

using T2 = ordered_value_type<0>;
template
  bool is_subseteq_of(std::unordered_set<T2> const &, std::unordered_set<T2> const &);

} // namespace FlexFlow
