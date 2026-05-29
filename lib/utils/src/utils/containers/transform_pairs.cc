#include "utils/containers/transform_pairs.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;
using Out = value_type<2>;
using F = std::function<Out(L const &, R const &)>;

template std::vector<Out> transform_pairs(std::vector<std::pair<L, R>> const &,
                                          F &&);

template std::unordered_set<Out>
    transform_pairs(std::unordered_set<std::pair<L, R>> const &, F &&);

} // namespace FlexFlow
