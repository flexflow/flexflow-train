#include "utils/containers/inplace_filter.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using Elem = value_type<0>;
using O_Elem = ordered_value_type<0>;

template void inplace_filter(std::vector<Elem> &,
                             std::function<bool(Elem const &)> const &);

template void inplace_filter(std::unordered_set<Elem> &,
                             std::function<bool(Elem const &)> const &);

template void inplace_filter(std::set<O_Elem> &,
                             std::function<bool(O_Elem const &)> const &);

using V = value_type<0>;

template void
    inplace_filter(std::unordered_map<Elem, V> &s,
                   std::function<bool(std::pair<Elem, V> const &)> const &);

template void
    inplace_filter(std::map<O_Elem, V> &s,
                   std::function<bool(std::pair<O_Elem, V> const &)> const &);

} // namespace FlexFlow
