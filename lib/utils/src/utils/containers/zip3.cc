#include "utils/containers/zip3.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using A1 = value_type<0>;
using B1 = value_type<1>;
using C1 = value_type<2>;

template std::vector<std::tuple<A1, B1, C1>> zip3(std::vector<A1> const &,
                                                  std::vector<B1> const &,
                                                  std::vector<C1> const &);

} // namespace FlexFlow
