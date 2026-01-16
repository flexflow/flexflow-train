#include "test/utils/doctest/fmt/monostate.h"

namespace doctest {

String StringMaker<std::monostate>::convert(std::monostate const &m) {
  return toString(fmt::to_string(m));
}

} // namespace doctest
