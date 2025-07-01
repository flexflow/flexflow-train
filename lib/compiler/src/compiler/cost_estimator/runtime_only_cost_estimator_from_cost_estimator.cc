#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/runtime_only_op_cost_metrics.h"

namespace FlexFlow {

RuntimeOnlyCostEstimatorFromCostEstimator::RuntimeOnlyCostEstimatorFromCostEstimator(
  CostEstimator const &cost_estimator)
  : cost_estimator(cost_estimator)
{ }

RuntimeOnlyOpCostMetrics RuntimeOnlyCostEstimatorFromCostEstimator::estimate_cost(RuntimeOnlyOpCostEstimateKey const &runtime_only) const {
  OptimizerAttrs fake_optimizer_attrs = OptimizerAttrs{
    SGDOptimizerAttrs{
      /*lr=*/0.0,
      /*momentum=*/0.0,
      /*nesterov=*/false,
      /*weight_decay=*/0.0,
    },
  };

  OpCostEstimateKey op_cost_estimate_key = 
    make_op_cost_estimate_key_from_runtime_only(runtime_only, fake_optimizer_attrs);

  OpCostMetrics op_cost_metrics = this->cost_estimator.estimate_cost(op_cost_estimate_key);

  return runtime_only_from_op_cost_metrics(op_cost_metrics);
}

milliseconds_t RuntimeOnlyCostEstimatorFromCostEstimator::estimate_cost(TensorSetMovement const &tensor_set_movement) const {
  return this->cost_estimator.estimate_cost(tensor_set_movement);
}

RuntimeOnlyCostEstimator runtime_only_cost_estimator_from_cost_estimator(CostEstimator const &cost_estimator) {
  return RuntimeOnlyCostEstimator::create<RuntimeOnlyCostEstimatorFromCostEstimator>(cost_estimator);
}

} // namespace FlexFlow
