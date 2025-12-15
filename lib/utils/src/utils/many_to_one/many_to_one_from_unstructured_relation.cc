#include "utils/many_to_one/many_to_one_from_unstructured_relation.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  ManyToOne<L, R> many_to_one_from_unstructured_relation(
    std::unordered_set<std::pair<L, R>> const &);

} // namespace FlexFlow
