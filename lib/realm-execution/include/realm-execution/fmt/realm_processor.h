#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_FMT_REALM_PROCESSOR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_FMT_REALM_PROCESSOR_H

#include "realm-execution/realm.h"
#include <fmt/format.h>
#include <utility>

namespace fmt {

template <typename Char>
struct formatter<
    ::FlexFlow::Realm::Processor,
    Char,
    std::enable_if_t<!detail::has_format_as<::FlexFlow::Realm::Processor>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::Realm::Processor const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {

    std::string result = fmt::format("<Processor {}>", m.id);

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

std::ostream &operator<<(std::ostream &s, ::FlexFlow::Realm::Processor const &m);

} // namespace FlexFlow

#endif
