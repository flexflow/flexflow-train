#include "utils/unordered_bidict/algorithms/merge_disjoint_unordered_bidicts.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template unordered_bidict<K, V> merge_disjoint_unordered_bidicts(
    std::vector<unordered_bidict<K, V>> const &);
template unordered_bidict<K, V>
    merge_disjoint_unordered_bidicts(std::set<unordered_bidict<K, V>> const &);

} // namespace FlexFlow
