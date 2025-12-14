#include "utils/containers/try_get_one_of.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::optional<T> try_get_one_of(std::unordered_set<T> const &);

using R = ordered_value_type<0>;

template
  std::optional<R> try_get_one_of(std::set<R> const &);


} // namespace FlexFlow
