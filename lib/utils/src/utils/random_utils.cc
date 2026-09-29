#include "utils/random_utils.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using G = std::mt19937;

template float randf(G &);

template T select_random(G &, std::vector<T> const &);

template T
    select_random(G &, std::vector<T> const &, std::vector<float> const &);

} // namespace FlexFlow
