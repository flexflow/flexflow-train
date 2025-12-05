#include "utils/containers/zip_values_strict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V1 = value_type<1>;
using V2 = value_type<2>;

template
  std::unordered_map<K, std::pair<V1, V2>> zip_values_strict(
    std::unordered_map<K, V1> const &,
    std::unordered_map<K, V2> const &);


} // namespace FlexFlow
