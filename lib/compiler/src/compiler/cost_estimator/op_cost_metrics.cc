#include "compiler/cost_estimator/op_cost_metrics.h"

namespace FlexFlow {

OpCostMetrics
  make_op_cost_metrics_from_runtime_only(
    RuntimeOnlyOpCostMetrics const &runtime_only,
    num_bytes_t const &memory_usage) {

  return OpCostMetrics{
    /*forward_runtime=*/runtime_only.forward_runtime,
    /*backward_runtime=*/runtime_only.backward_runtime,
    /*memory_usage=*/memory_usage,
  };
}

} // namespace FlexFlow
