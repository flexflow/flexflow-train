#include "utils/containers/repeat.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Out = value_type<0>;
using F = std::function<Out()>;

template std::vector<Out> repeat(nonnegative_int, F const &);

} // namespace FlexFlow
