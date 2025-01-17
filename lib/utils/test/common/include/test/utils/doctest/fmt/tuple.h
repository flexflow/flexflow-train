#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_TUPLE_H

#include "utils/fmt/tuple.h"
#include <doctest/doctest.h>

namespace doctest {

template <typename ...Ts>
struct StringMaker<std::tuple<Ts...>> {
  static String convert(std::tuple<Ts...> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
