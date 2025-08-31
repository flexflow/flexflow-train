#include "utils/containers/extend.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using C = std::unordered_multiset<T>;

template
  void extend(std::vector<T> &, C const &);

template
  void extend(std::unordered_set<T> &, C const &);

template
  void extend(std::unordered_multiset<T> &, C const &);

using T2 = ordered_value_type<0>;
using C2 = std::unordered_multiset<T2>;

template
  void extend(std::set<T2> &, C2 const &);

template
  void extend(std::multiset<T2> &, C2 const &);

} // namespace FlexFlow
