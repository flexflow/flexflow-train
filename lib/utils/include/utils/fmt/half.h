#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_HALF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_HALF_H

#include <fmt/format.h>
#include "utils/half.h"

namespace fmt {

template <typename Char>
struct formatter<::half, Char>
    : formatter<float> {
  template <typename FormatContext>
  auto format(::half const &h, FormatContext &ctx)
      -> decltype(ctx.out()) {

    return formatter<float>::format(h, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

std::ostream &operator<<(std::ostream &, ::half);

} // namespace FlexFlow

#endif
