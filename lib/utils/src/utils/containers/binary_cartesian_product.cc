#include "utils/containers/binary_cartesian_product.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using A = value_type<0>;
using B = value_type<1>;

template
  std::unordered_set<std::pair<A, B>>
    binary_cartesian_product(std::unordered_set<A> const &,
                             std::unordered_set<B> const &);
template
  std::unordered_set<std::pair<A, A>>
    binary_cartesian_product(std::unordered_set<A> const &,
                             std::unordered_set<A> const &);

} // namespace FlexFlow
