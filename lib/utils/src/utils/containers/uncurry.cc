#include "utils/containers/uncurry.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using Result = value_type<2>;
using F = std::function<Result(T1 const &, T2 const &)>;

template 
  std::function<Result(std::pair<T1, T2> const &)> uncurry(F &&);

} // namespace FlexFlow
