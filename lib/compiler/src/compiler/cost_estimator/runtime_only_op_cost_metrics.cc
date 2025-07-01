#include "compiler/cost_estimator/runtime_only_op_cost_metrics.h"

namespace FlexFlow {

RuntimeOnlyOpCostMetrics
  runtime_only_from_op_cost_metrics(OpCostMetrics const &op_cost_metrics) {
  
  return RuntimeOnlyOpCostMetrics{
    /*forward_runtime=*/op_cost_metrics.forward_runtime,
    /*backward_runtime=*/op_cost_metrics.backward_runtime,
  };
}

} // namespace FlexFlow
