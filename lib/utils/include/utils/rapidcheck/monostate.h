#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_MONOSTATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_MONOSTATE_H

#include <rapidcheck.h>
#include <variant>

namespace rc {

template <>
struct Arbitrary<std::monostate> {
  static Gen<std::monostate> arbitrary();
};

} // namespace rc

#endif
