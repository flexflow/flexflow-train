#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_FMT_REALM_PROCESSOR_KIND_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_FMT_REALM_PROCESSOR_KIND_H

#include "realm-execution/realm.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <utility>

namespace fmt {

template <typename Char>
struct formatter<::FlexFlow::Realm::Processor::Kind, Char>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::Realm::Processor::Kind const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {

    std::string result;
    switch (m) {
      case ::FlexFlow::Realm::Processor::Kind::NO_KIND: {
        result = "<Processor::Kind::NO_KIND>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::TOC_PROC: {
        // Throughput core
        result = "<Processor::Kind::TOC_PROC>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::LOC_PROC: {
        // Latency core
        result = "<Processor::Kind::LOC_PROC>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::UTIL_PROC: {
        // Utility core
        result = "<Processor::Kind::UTIL_PROC>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::IO_PROC: {
        // I/O core
        result = "<Processor::Kind::IO_PROC>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::PROC_GROUP: {
        // Processor group
        result = "<Processor::Kind::PROC_GROUP>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::PROC_SET: {
        // Set of Processors for OpenMP/Kokkos etc.
        result = "<Processor::Kind::PROC_SET>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::OMP_PROC: {
        // OpenMP (or similar) thread pool
        result = "<Processor::Kind::OMP_PROC>";
        break;
      }
      case ::FlexFlow::Realm::Processor::Kind::PY_PROC: {
        // Python interpreter
        result = "<Processor::Kind::PY_PROC>";
        break;
      }
      default:
        PANIC("Unknown Processor::Kind {}", static_cast<int>(m));
    };

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::Realm::Processor::Kind const &);

} // namespace FlexFlow

#endif
