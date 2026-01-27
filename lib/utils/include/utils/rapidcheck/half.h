#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_HALF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_HALF_H

#include "utils/half.h"
#include <rapidcheck.h>

namespace rc {

template <>
struct Arbitrary<::half> {
  static Gen<::half> arbitrary();
};

} // namespace rc

#endif
