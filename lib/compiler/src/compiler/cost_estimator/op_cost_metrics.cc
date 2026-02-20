#include "compiler/cost_estimator/op_cost_metrics.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

bool is_pareto_optimal_in(OpCostMetrics const &m,
                          std::unordered_set<OpCostMetrics> const &others) {
  return all_of(others, [&](OpCostMetrics const &other) {
    return m.forward_runtime <= other.forward_runtime ||
           m.backward_runtime <= other.backward_runtime ||
           m.memory_usage <= other.memory_usage;
  });
}

OpCostMetrics make_op_cost_metrics_from_runtime_only(
    RuntimeOnlyOpCostMetrics const &runtime_only,
    num_bytes_t const &memory_usage) {

  return OpCostMetrics{
      /*forward_runtime=*/runtime_only.forward_runtime,
      /*backward_runtime=*/runtime_only.backward_runtime,
      /*memory_usage=*/memory_usage,
  };
}

} // namespace FlexFlow
