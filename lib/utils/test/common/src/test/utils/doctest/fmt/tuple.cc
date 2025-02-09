#include "test/utils/doctest/fmt/tuple.h"
#include "utils/archetypes/value_type.h"

using ::FlexFlow::value_type;

using A = value_type<0>;
using B = value_type<1>;
using C = value_type<2>;

namespace doctest {

template struct StringMaker<std::tuple<A, B, C>>;

} // namespace doctest
