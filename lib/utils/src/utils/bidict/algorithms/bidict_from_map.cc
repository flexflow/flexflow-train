#include "utils/bidict/algorithms/bidict_from_map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

template
  bidict<value_type<0>, value_type<1>> bidict_from_map(
     std::unordered_map<value_type<0>, value_type<1>> const &);

template
  bidict<ordered_value_type<0>, value_type<1>> bidict_from_map(
     std::map<ordered_value_type<0>, value_type<1>> const &);

} // namespace FlexFlow
