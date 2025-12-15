#include "utils/one_to_many/unstructured_relation_from_one_to_many.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  std::unordered_set<std::pair<L, R>>
    unstructured_relation_from_one_to_many(
      OneToMany<L, R> const &);

} // namespace FlexFlow
