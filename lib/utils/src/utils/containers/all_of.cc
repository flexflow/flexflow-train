#include "utils/containers/all_of.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"
#include <unordered_set>
#include <set>

namespace FlexFlow {

using T1 = value_type<0>;
using F1 = std::function<bool(T1)>;

template
  bool all_of(std::vector<T1> const &, F1 &&);
template
  bool all_of(std::unordered_set<T1> const &, F1 &&);
template
  bool all_of(std::unordered_multiset<T1> const &, F1 &&);

using T2 = ordered_value_type<0>;
using F2 = std::function<bool(T2)>;

template
  bool all_of(std::set<T2> const &, F2 &&);
template
  bool all_of(std::multiset<T2> const &, F2 &&);

using K3 = value_type<0>;
using V3 = value_type<1>;
using F3 = std::function<bool(K3, V3)>;

template
  bool all_of(std::unordered_map<K3, V3> const &, F3 &&);

using K4 = ordered_value_type<0>;
using V4 = ordered_value_type<1>;
using F4 = std::function<bool(K4, V4)>;

template
  bool all_of(std::map<K4, V4> const &, F4 &&);


bool all_of(std::vector<bool> const &v) {
  for (bool v : v) {
    if (!v) {
      return false;
    }
  }

  return true;
}

} // namespace FlexFlow
