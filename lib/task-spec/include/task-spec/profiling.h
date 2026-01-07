#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PROFILING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PROFILING_H

#include "kernels/profiling.h"
#include <spdlog/spdlog.h>

namespace FlexFlow {

enum class EnableProfiling { YES, NO };

template <typename F, typename... Ts, typename Str>
std::optional<milliseconds_t> profile(F const &f,
                                      ProfilingSettings profiling,
                                      DeviceType device_type,
                                      Str s,
                                      Ts &&...ts) {
  std::optional<milliseconds_t> elapsed = profiling_wrapper<F, Ts...>(
      f, profiling, device_type, std::forward<Ts>(ts)...);
  if (elapsed.has_value()) {
    spdlog::debug(s, elapsed.value());
  }
  return elapsed;
}

} // namespace FlexFlow

#endif
