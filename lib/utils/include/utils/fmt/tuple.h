#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_TUPLE_H

#include "utils/check_fmtable.h"
#include <fmt/format.h>
#include <tuple>
#include <cassert>
#include <vector>
#include "utils/join_strings.h"
#include "utils/tuple/visit.h"

namespace fmt {

template <typename... Ts, typename Char>
struct formatter<std::tuple<Ts...>, Char>
    : formatter<std::string> {

  template <typename FormatContext>
  auto format(std::tuple<Ts...> const &t, FormatContext &ctx) const
      -> decltype(ctx.out()) {

    std::vector<std::string> stringified_elements;
    ::FlexFlow::visit_tuple(t, [&](auto const &element) -> void { stringified_elements.push_back(fmt::to_string(element)); });

    return formatter<std::string>::format("{" + ::FlexFlow::join_strings(stringified_elements, ", ") + "}", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename ...Ts>
std::ostream &operator<<(std::ostream &s, std::tuple<Ts...> const &t) {
  return (s << fmt::to_string(t));
}

} // namespace FlexFlow

#endif
