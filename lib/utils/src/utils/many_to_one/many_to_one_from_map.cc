#include "utils/many_to_one/many_to_one_from_map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L1 = value_type<0>;
using R1 = value_type<1>;

template
  ManyToOne<L1, R1>
    many_to_one_from_map(
      std::unordered_map<L1, R1> const &);

using L2 = ordered_value_type<0>;
using R2 = ordered_value_type<1>;

template
  ManyToOne<L2, R2>
    many_to_one_from_map(
      std::map<L2, R2> const &);

} // namespace FlexFlow
