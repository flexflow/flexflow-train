#include "test/utils/doctest/fmt/tuple.h"

namespace doctest {

template
  struct StringMaker<std::tuple<int, bool, int>>;

} // namespace doctest
