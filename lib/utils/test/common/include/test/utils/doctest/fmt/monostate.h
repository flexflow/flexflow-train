#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_MONOSTATE_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_MONOSTATE_H

#include "utils/fmt/monostate.h"
#include <doctest/doctest.h>

namespace doctest {

template <>
struct StringMaker<std::monostate> {
  static String convert(std::monostate const &);
};

} // namespace 

#endif
