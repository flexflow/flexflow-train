#include "utils/containers/recurse_n.h"
#include "utils/archetypes/value_type.h"
#include <functional>

namespace FlexFlow {

using T = value_type<0>;
using F = std::function<T(T)>; // F :: T -> T

template T recurse_n(F const &f, int n, T const &initial_value);

} // namespace FlexFlow
