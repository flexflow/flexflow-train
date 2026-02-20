#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_RUNTIME_ONLY_COST_ESTIMATOR_H

#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/runtime_only_op_cost_metrics.dtg.h"
#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <vector>

namespace FlexFlow {

struct IRuntimeOnlyCostEstimator {
  virtual RuntimeOnlyOpCostMetrics
      estimate_cost(RuntimeOnlyOpCostEstimateKey const &) const = 0;
  virtual milliseconds_t estimate_cost(TensorSetMovement const &) const = 0;

  IRuntimeOnlyCostEstimator() = default;
  IRuntimeOnlyCostEstimator(IRuntimeOnlyCostEstimator const &) = delete;
  IRuntimeOnlyCostEstimator &
      operator=(IRuntimeOnlyCostEstimator const &) = delete;

  virtual ~IRuntimeOnlyCostEstimator() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IRuntimeOnlyCostEstimator);

struct RuntimeOnlyCostEstimator {
  RuntimeOnlyOpCostMetrics
      estimate_cost(RuntimeOnlyOpCostEstimateKey const &) const;
  milliseconds_t estimate_cost(TensorSetMovement const &m) const;

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IRuntimeOnlyCostEstimator, T>::value,
      RuntimeOnlyCostEstimator>::type
      create(Args &&...args) {
    return RuntimeOnlyCostEstimator(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  RuntimeOnlyCostEstimator(
      std::shared_ptr<IRuntimeOnlyCostEstimator> implementation_ptr);

private:
  std::shared_ptr<IRuntimeOnlyCostEstimator> implementation_ptr;
};

} // namespace FlexFlow

#endif
