#include "utils/bidict/algorithms/binary_merge_disjoint_bidicts.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template bidict<K, V> binary_merge_disjoint_bidicts(bidict<K, V> const &,
                                                    bidict<K, V> const &);

} // namespace FlexFlow
