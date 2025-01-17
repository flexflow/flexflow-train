#include "utils/containers/scanr1.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include <set>

namespace FlexFlow {

using T = value_type<0>;
using C = std::vector<T>;
using F = std::function<T(T, T)>;

template
  std::vector<T> scanr1(std::vector<T> const &, F &&);

using T2 = ordered_value_type<0>;
using F2 = std::function<T2(T2, T2)>;

template
  std::vector<T2> scanr1(std::set<T2> const &, F2 &&);

} // namespace FlexFlow
