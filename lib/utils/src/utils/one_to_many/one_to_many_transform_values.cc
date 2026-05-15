#include "utils/one_to_many/one_to_many_transform_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R1 = value_type<1>;
using R2 = value_type<2>;
using F = std::function<R2(R1 const &)>;

template OneToMany<L, R2> one_to_many_transform_values(OneToMany<L, R1> const &,
                                                       F);

} // namespace FlexFlow
