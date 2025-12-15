#include "utils/many_to_one/unstructured_relation_from_many_to_one.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  std::unordered_set<std::pair<L, R>> 
    unstructured_relation_from_many_to_one(
      ManyToOne<L, R> const &);

} // namespace FlexFlow
