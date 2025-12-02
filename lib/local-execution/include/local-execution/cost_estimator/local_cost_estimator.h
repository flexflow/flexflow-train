#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_ESTIMATOR_LOCAL_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_ESTIMATOR_LOCAL_COST_ESTIMATOR_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "pcg/machine_interconnect_specification.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_config.dtg.h"

namespace FlexFlow {

struct LocalCostEstimator : public ICostEstimator {
  explicit LocalCostEstimator(RuntimeArgConfig const &,
                              MachineInterconnectSpecification const &,
                              DeviceType);

  LocalCostEstimator(LocalCostEstimator const &) = delete;
  LocalCostEstimator(LocalCostEstimator &&) = delete;
  ~LocalCostEstimator() = default;

  OpCostMetrics estimate_cost(OpCostEstimateKey const &) const override;

  milliseconds_t estimate_cost(TensorSetMovement const &) const override;

private:
  RuntimeArgConfig runtime_arg_config;
  MachineInterconnectSpecification interconnect_specification;
  DeviceType device_type;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCostEstimator);

CostEstimator get_local_cost_estimator(RuntimeArgConfig const &);

} // namespace FlexFlow

#endif
