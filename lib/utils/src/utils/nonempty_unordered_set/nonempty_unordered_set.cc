#include "utils/nonempty_unordered_set/nonempty_unordered_set.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct nonempty_unordered_set<T>;

} // namespace FlexFlow
