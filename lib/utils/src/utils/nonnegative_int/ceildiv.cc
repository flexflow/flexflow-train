#include "utils/nonnegative_int/ceildiv.h"
#include "utils/exception.h"

namespace FlexFlow {

nonnegative_int ceildiv(nonnegative_int numerator,
                        nonnegative_int denominator) {
  if (denominator == 0) {
    throw mk_runtime_error(fmt::format(
        "ceildiv expected denominator != 0, but received {}", denominator));
  }

  int n = numerator.unwrap_nonnegative();
  int d = denominator.unwrap_nonnegative();

  int result = (n + d - 1) / d;
  return nonnegative_int{result};
}

} // namespace FlexFlow
