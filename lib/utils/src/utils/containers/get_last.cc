#include "utils/containers/get_last.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template T const &get_last(std::vector<T> const &);

} // namespace FlexFlow
