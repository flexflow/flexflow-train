#include "utils/containers/flatmap.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using In = value_type<0>;
using Out = value_type<1>;
using F = std::function<std::vector<Out>(In)>;

template
  std::vector<Out> flatmap(std::vector<In> const &, F &&);

template
  std::unordered_set<Out> flatmap(std::unordered_set<In> const &, F &&);

template
  std::unordered_multiset<Out> flatmap(std::unordered_multiset<In> const &, F &&);

using In2 = ordered_value_type<0>;
using Out2 = ordered_value_type<1>;
using F2 = std::function<std::vector<Out2>(In2)>;

template
  std::set<Out2> flatmap(std::set<In2> const &, F2 &&);

template
  std::multiset<Out2> flatmap(std::multiset<In2> const &, F2 &&);

using InK = value_type<2>;
using InV = value_type<3>;
using OutK = value_type<4>;
using OutV = value_type<5>;
using F3 = std::function<std::unordered_map<OutK, OutV>(InK, InV)>;

template
  std::unordered_map<OutK, OutV> flatmap(std::unordered_map<InK, InV> const &, F3 &&);

using F4 = std::function<std::string(char)>;

template
  std::string flatmap(std::string const &, F4 &&f);


} // namespace FlexFlow
