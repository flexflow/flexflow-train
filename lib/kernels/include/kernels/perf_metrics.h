#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_PERF_METRICS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_PERF_METRICS_H

#include "kernels/perf_metrics.dtg.h"
#include <fmt/format.h>

namespace FlexFlow {

float get_throughput(PerfMetrics const &);
float get_accuracy(PerfMetrics const &);

PerfMetrics update(PerfMetrics const &, PerfMetrics const &);
PerfMetrics apply_scale(PerfMetrics const &, float scale);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::PerfMetrics> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::PerfMetrics const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    auto out = fmt::memory_buffer();
    fmt::format_to(std::back_inserter(out), "PerfMetrics[");
    if (m.train_correct.has_value()) {
      fmt::format_to(std::back_inserter(out),
                     " accuracy={:.2f}%",
                     100.0 * get_accuracy(m));
    }
    if (m.cce_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " cce={:.2f}", m.cce_loss.value());
    }
    if (m.sparse_cce_loss.has_value()) {
      fmt::format_to(std::back_inserter(out),
                     " sparse_cce={:.2f}",
                     m.sparse_cce_loss.value());
    }
    if (m.mse_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " mse={:.2f}", m.mse_loss.value());
    }
    if (m.rmse_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " rmse={:.2f}", m.rmse_loss.value());
    }
    if (m.mae_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " mae={:.2f}", m.mae_loss.value());
    }
    fmt::format_to(
        std::back_inserter(out), "throughput={:.2f}", get_throughput(m));
    return formatter<std::string>::format(fmt::to_string(out), ctx);
  }
};

} // namespace fmt

#endif
