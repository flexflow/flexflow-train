#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template bidict<K, V> merge_disjoint_bidicts(std::set<bidict<K, V>> const &);

} // namespace FlexFlow
