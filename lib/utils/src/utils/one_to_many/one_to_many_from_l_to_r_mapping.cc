#include "utils/one_to_many/one_to_many_from_l_to_r_mapping.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  OneToMany<L, R> one_to_many_from_l_to_r_mapping(std::unordered_map<L, std::unordered_set<R>> const &);

} // namespace FlexFlow
