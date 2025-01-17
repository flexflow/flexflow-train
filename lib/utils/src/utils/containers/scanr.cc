#include "utils/containers/scanr.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include <set>

namespace FlexFlow {

using T = value_type<0>;
using C = std::vector<T>;
using F = std::function<T(T, T)>;

template
  std::vector<T> scanr(std::vector<T> const &, T, F &&);

using T2 = ordered_value_type<0>;
using F2 = std::function<T2(T2, T2)>;

template
  std::vector<T2> scanr(std::set<T2> const &, T2, F2 &&);


} // namespace FlexFlow
