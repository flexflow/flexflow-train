#include "utils/containers/zip_strict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template std::vector<std::pair<L, R>> zip_strict(std::vector<L> const &,
                                                 std::vector<R> const &);

} // namespace FlexFlow
