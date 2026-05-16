#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template bidict<K, V> merge_disjoint_bidicts(std::vector<bidict<K, V>> const &);

} // namespace FlexFlow
