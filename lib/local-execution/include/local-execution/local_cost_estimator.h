#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_COST_ESTIMATOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_COST_ESTIMATOR_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"

namespace FlexFlow {

struct LocalCostEstimator : public ICostEstimator {
  LocalCostEstimator(RuntimeArgConfig const &, OptimizerAttrs const &);

  LocalCostEstimator(LocalCostEstimator const &) = delete;
  LocalCostEstimator(LocalCostEstimator &&) = delete;
  ~LocalCostEstimator() = default;

  OpCostMetrics estimate_cost(OpCostEstimateKey const &) const override;

  milliseconds_t estimate_cost(TensorSetMovement const &) const override;
private:
  RuntimeArgConfig runtime_arg_config;
  OptimizerAttrs optimizer_attrs;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCostEstimator);

CostEstimator get_local_cost_estimator(RuntimeArgConfig const &, OptimizerAttrs const &);

} // namespace FlexFlow

#endif
