#include "op-attrs/ff_ordered/zip.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;

template FFOrdered<std::pair<T1, T2>> zip(FFOrdered<T1> const &,
                                          FFOrdered<T2> const &);

} // namespace FlexFlow
