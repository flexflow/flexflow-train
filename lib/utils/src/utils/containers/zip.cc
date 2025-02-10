#include "utils/containers/zip.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L1 = value_type<0>;
using R1 = value_type<1>;

template std::vector<std::pair<L1, R1>> zip(std::vector<L1> const &,
                                            std::vector<R1> const &);

} // namespace FlexFlow
