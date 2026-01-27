#include "kernels/perf_metrics.h"

namespace FlexFlow {

float get_throughput(PerfMetrics const &m) {
  return m.train_all / (m.current_time - m.start_time);
}

float get_accuracy(PerfMetrics const &m) {
  return static_cast<float>(m.train_correct.value()) / m.train_all;
}

PerfMetrics update(PerfMetrics const &lhs, PerfMetrics const &rhs) {
  PerfMetrics out(lhs);

  auto update_val = [](std::optional<float> &l, std::optional<float> const &r) {
    if (l.has_value()) {
      l.value() += r.value();
    }
  };

  out.train_all += rhs.train_all;
  if (out.train_correct.has_value()) {
    out.train_correct.value() += rhs.train_correct.value();
  }
  update_val(out.cce_loss, rhs.cce_loss);
  update_val(out.sparse_cce_loss, rhs.sparse_cce_loss);
  update_val(out.mse_loss, rhs.mse_loss);
  update_val(out.rmse_loss, rhs.rmse_loss);
  update_val(out.mae_loss, rhs.mae_loss);
  out.current_time = rhs.current_time;

  return out;
}

PerfMetrics apply_scale(PerfMetrics const &pm, float scale) {
  PerfMetrics out(pm);

  auto scale_val = [&](std::optional<float> &l) {
    if (l.has_value()) {
      l.value() *= scale;
    }
  };

  scale_val(out.cce_loss);
  scale_val(out.sparse_cce_loss);
  scale_val(out.mse_loss);
  scale_val(out.rmse_loss);
  scale_val(out.mae_loss);

  return out;
}

} // namespace FlexFlow
