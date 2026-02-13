#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_FMT_INSTANCE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_FMT_INSTANCE_H

#include "realm-execution/realm.h"
#include "utils/check_fmtable.h"
#include <fmt/format.h>
#include <utility>

namespace fmt {

template <typename Char>
struct formatter<::FlexFlow::Realm::RegionInstance,
                 Char,
                 std::enable_if_t<!detail::has_format_as<
                     ::FlexFlow::Realm::RegionInstance>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::Realm::RegionInstance const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result = fmt::format("<RegionInstance {}>", m.id);

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::Realm::RegionInstance const &m);

} // namespace FlexFlow

#endif
