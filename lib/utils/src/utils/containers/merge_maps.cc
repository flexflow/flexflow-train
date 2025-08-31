#include "utils/containers/merge_maps.h"
#include "utils/archetypes/value_type.h"
#include "utils/hash/unordered_map.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template void merge_in_map(std::unordered_map<K, V> const &,
                           std::unordered_map<K, V> &);

template std::unordered_map<K, V>
    merge_disjoint_maps(std::unordered_map<K, V> const &,
                        std::unordered_map<K, V> const &);

template std::unordered_map<K, V>
    merge_map_left_dominates(std::unordered_map<K, V> const &,
                             std::unordered_map<K, V> const &);

template std::unordered_map<K, V>
    merge_map_right_dominates(std::unordered_map<K, V> const &,
                              std::unordered_map<K, V> const &);

using C = std::vector<std::unordered_map<K, V>>;

template std::unordered_map<K, V> merge_maps(C const &);
} // namespace FlexFlow
