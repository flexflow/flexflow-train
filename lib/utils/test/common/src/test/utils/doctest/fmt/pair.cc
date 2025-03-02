#include "test/utils/doctest/fmt/pair.h"
#include "utils/archetypes/value_type.h"

using ::FlexFlow::value_type;

using L = value_type<0>;
using R = value_type<1>;

namespace doctest {

template struct StringMaker<std::pair<L, R>>;

} // namespace doctest
