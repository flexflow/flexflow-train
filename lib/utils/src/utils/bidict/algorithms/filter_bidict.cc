#include "utils/bidict/algorithms/filter_bidict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;
using F = std::function<bool(L const &, R const &)>;

template bidict<L, R> filter_bidict(bidict<L, R> const &, F &&);

} // namespace FlexFlow
