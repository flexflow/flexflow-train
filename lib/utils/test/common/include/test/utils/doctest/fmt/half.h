#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_HALF_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_FMT_HALF_H

#include "utils/half.h"
#include <doctest/doctest.h>

namespace doctest {

template <>
struct StringMaker<::half> {
  static String convert(::half const &);
};

} // namespace doctest

#endif
