#include "utils/containers/filter.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using VT0 = value_type<0>;
using VT1 = value_type<1>;
using OVT0 = ordered_value_type<0>;

template
  std::vector<VT0> filter(std::vector<VT0> const &,
                          std::function<bool(VT0 const &)> const &);

template
  std::unordered_set<VT0> filter(std::unordered_set<VT0> const &,
                                 std::function<bool(VT0 const &)> const &);

template
  std::unordered_map<VT0, VT1>
    filter(std::unordered_map<VT0, VT1> const &,
           std::function<bool(std::pair<VT0, VT1> const &)> const &);

template
  std::set<OVT0> filter(
    std::set<OVT0> const &,
    std::function<bool(OVT0 const &)> const &);

template
  std::map<OVT0, VT1>
    filter(std::map<OVT0, VT1> const &,
           std::function<bool(std::pair<OVT0, VT1> const &)> const &);

template
  std::multiset<OVT0>
    filter(std::multiset<OVT0> const &,
           std::function<bool(OVT0 const &)> const &);

template
  std::unordered_multiset<VT0>
    filter(std::unordered_multiset<VT0> const &,
           std::function<bool(VT0 const &)> const &);


} // namespace FlexFlow
