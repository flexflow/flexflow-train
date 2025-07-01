#ifndef _FLEXFLOW_LIB_COMPILER_TEST_SRC_INTERNAL_RUNTIME_ONLY_COST_ESTIMATOR_FOR_TEST_H
#define _FLEXFLOW_LIB_COMPILER_TEST_SRC_INTERNAL_RUNTIME_ONLY_COST_ESTIMATOR_FOR_TEST_H

#include "compiler/cost_estimator/runtime_only_cost_estimator.h"

namespace FlexFlow {

RuntimeOnlyCostEstimator make_fake_runtime_only_cost_estimator(
    std::function<RuntimeOnlyOpCostMetrics(RuntimeOnlyOpCostEstimateKey const &)> const
        &get_operator_cost,
    std::function<milliseconds_t(TensorSetMovement const &)> const
        &get_communication_cost);

RuntimeOnlyCostEstimator make_fake_runtime_only_cost_estimator(
    std::unordered_map<RuntimeOnlyOpCostEstimateKey, RuntimeOnlyOpCostMetrics> const &op_cost_map,
    std::unordered_map<TensorSetMovement, milliseconds_t> const &comm_cost_map);


RuntimeOnlyCostEstimator make_fake_constant_runtime_only_cost_estimator(milliseconds_t forward_op_cost,
                                                           milliseconds_t backward_op_cost,
                                                           milliseconds_t comm_cost);


} // namespace FlexFlow

#endif
