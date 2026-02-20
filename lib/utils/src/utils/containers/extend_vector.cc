#include "utils/containers/extend_vector.h"
#include "utils/archetypes/value_type.h"
#include <unordered_set>

namespace FlexFlow {

using T = value_type<0>;

template void extend_vector(std::vector<T> &, std::unordered_set<T> const &);

} // namespace FlexFlow
