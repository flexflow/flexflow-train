#include "utils/singular_or_variadic.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using Out = value_type<1>;

template
  SingularOrVariadic<Out> transform_singular_or_variadic(
    SingularOrVariadic<T> const &, std::function<Out(T const &)> &&);

using T1 = value_type<2>;
using T2 = value_type<3>;

template
  SingularOrVariadic<std::pair<T1, T2>> zip_strict_singular_or_variadic(
      SingularOrVariadic<T1> const &,
      SingularOrVariadic<T2> const &);

template
  SingularOrVariadic<Out> zip_strict_singular_or_variadic_with(
      SingularOrVariadic<T1> const &,
      SingularOrVariadic<T2> const &,
      std::function<Out(T1 const &, T2 const &)> &&);

} // namespace FlexFlow
