#include "utils/containers/map_from_keys_and_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K1 = value_type<0>;
using V1 = value_type<1>;

template
  std::unordered_map<K1, V1>
      map_from_keys_and_values(std::vector<K1> const &,
                               std::vector<V1> const &);

} // namespace FlexFlow
