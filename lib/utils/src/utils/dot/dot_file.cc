#include "utils/dot/dot_file.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template class DotFile<T>;

} // namespace FlexFlow
