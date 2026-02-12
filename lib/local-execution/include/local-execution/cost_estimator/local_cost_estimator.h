#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_ESTIMATOR_LOCAL_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_ESTIMATOR_LOCAL_COST_ESTIMATOR_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_interconnect_specification.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"

namespace FlexFlow {

struct LocalCostEstimator : public ICostEstimator {
  explicit LocalCostEstimator(
      MachineInterconnectSpecification const &interconnect_specification,
      Allocator &allocator,
      ProfilingSettings const &profiling_settings,
      device_handle_t const &device_handle,
      FFIterationConfig const &iteration_config,
      device_id_t device_idx);

  LocalCostEstimator(LocalCostEstimator const &) = delete;
  LocalCostEstimator(LocalCostEstimator &&) = delete;
  ~LocalCostEstimator() = default;

  OpCostMetrics estimate_cost(OpCostEstimateKey const &) const override;

  milliseconds_t estimate_cost(TensorSetMovement const &) const override;

private:
  MachineInterconnectSpecification interconnect_specification;
  Allocator allocator;
  ProfilingSettings profiling_settings;
  device_handle_t device_handle;
  FFIterationConfig iteration_config;
  device_id_t device_idx;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCostEstimator);

CostEstimator get_local_cost_estimator(
    MachineInterconnectSpecification const &interconnect_specification,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    FFIterationConfig const &iteration_config,
    device_id_t device_idx);

} // namespace FlexFlow

#endif
