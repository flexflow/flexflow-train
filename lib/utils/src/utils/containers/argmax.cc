#include "utils/containers/argmax.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include <functional>
#include <set>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

using T1 = value_type<0>;
using K1 = ordered_value_type<1>;
using F1 = std::function<K1(T1)>;

template T1 argmax(std::vector<T1> const &, F1 &&);
template T1 argmax(std::unordered_set<T1> const &, F1 &&);
template T1 argmax(std::unordered_multiset<T1> const &, F1 &&);

using T2 = ordered_value_type<0>;
using K2 = ordered_value_type<1>;
using F2 = std::function<K2(T2)>;

template T2 argmax(std::set<T2> const &, F2 &&);
template T2 argmax(std::multiset<T2> const &, F2 &&);

} // namespace FlexFlow
