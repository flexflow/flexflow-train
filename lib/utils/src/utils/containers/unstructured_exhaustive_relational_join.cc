#include "utils/containers/unstructured_exhaustive_relational_join.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using C = value_type<1>;
using R = value_type<2>;

template
  std::unordered_set<std::pair<L, R>>
    unstructured_exhaustive_relational_join(
      std::unordered_set<std::pair<L, C>> const &,
      std::unordered_set<std::pair<C, R>> const &);

} // namespace FlexFlow
