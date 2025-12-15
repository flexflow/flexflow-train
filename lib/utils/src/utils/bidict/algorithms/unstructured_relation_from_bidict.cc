#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  std::unordered_set<std::pair<L, R>>
    unstructured_relation_from_bidict(bidict<L, R> const &);

} // namespace FlexFlow
