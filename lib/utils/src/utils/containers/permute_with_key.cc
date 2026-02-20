#include "utils/containers/permute_with_key.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::vector<T> permute_with_key(int, std::vector<T> const &);

} // namespace FlexFlow
