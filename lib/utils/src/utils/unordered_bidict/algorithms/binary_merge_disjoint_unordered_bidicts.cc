#include "utils/unordered_bidict/algorithms/binary_merge_disjoint_unordered_bidicts.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template unordered_bidict<K, V> binary_merge_disjoint_unordered_bidicts(unordered_bidict<K, V> const &,
                                                    unordered_bidict<K, V> const &);

} // namespace FlexFlow
