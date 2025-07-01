#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_COST_ESTIMATOR_FROM_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_COST_ESTIMATOR_FROM_COST_ESTIMATOR_H

#include "compiler/cost_estimator/runtime_only_cost_estimator.h"
#include "compiler/cost_estimator/cost_estimator.h"

namespace FlexFlow {

struct RuntimeOnlyCostEstimatorFromCostEstimator final : public IRuntimeOnlyCostEstimator {
  RuntimeOnlyCostEstimatorFromCostEstimator() = delete;
  RuntimeOnlyCostEstimatorFromCostEstimator(
    CostEstimator const &cost_estimator); 

  RuntimeOnlyOpCostMetrics estimate_cost(RuntimeOnlyOpCostEstimateKey const &) const override;
  milliseconds_t estimate_cost(TensorSetMovement const &) const override;

private:
  CostEstimator cost_estimator;
};

RuntimeOnlyCostEstimator runtime_only_cost_estimator_from_cost_estimator(CostEstimator const &);

} // namespace FlexFlow

#endif
