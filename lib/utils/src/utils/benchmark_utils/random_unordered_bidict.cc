#include "utils/benchmark_utils/random_unordered_bidict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;
using FL = std::function<L(std::mt19937 &)>;
using FR = std::function<R(std::mt19937 &)>;

template unordered_bidict<L, R>
    random_unordered_bidict(std::mt19937 &, nonnegative_int, FL &&, FR &&);

} // namespace FlexFlow
