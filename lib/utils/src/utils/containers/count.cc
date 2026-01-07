#include "utils/containers/count.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using F = std::function<bool(T const &)>;

template nonnegative_int count(std::vector<T> const &, F const &);

} // namespace FlexFlow
