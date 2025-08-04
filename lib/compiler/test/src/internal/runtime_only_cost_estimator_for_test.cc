#include "internal/runtime_only_cost_estimator_for_test.h"
#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/cost_estimator/op_cost_metrics.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "internal/cost_estimator_for_test.h"

namespace FlexFlow {

RuntimeOnlyCostEstimator make_fake_runtime_only_cost_estimator(
    std::function<RuntimeOnlyOpCostMetrics(
        RuntimeOnlyOpCostEstimateKey const &)> const &get_operator_cost,
    std::function<milliseconds_t(TensorSetMovement const &)> const
        &get_communication_cost) {
  return runtime_only_cost_estimator_from_cost_estimator(
      make_fake_cost_estimator(
          [get_operator_cost](OpCostEstimateKey const &key) -> OpCostMetrics {
            RuntimeOnlyOpCostMetrics runtime_only_op_cost_metrics =
                get_operator_cost(runtime_only_from_op_cost_estimate_key(key));
            return make_op_cost_metrics_from_runtime_only(
                runtime_only_op_cost_metrics, 0_bytes);
          },
          get_communication_cost));
}

RuntimeOnlyCostEstimator make_fake_runtime_only_cost_estimator(
    std::unordered_map<RuntimeOnlyOpCostEstimateKey,
                       RuntimeOnlyOpCostMetrics> const &op_cost_map,
    std::unordered_map<TensorSetMovement, milliseconds_t> const
        &comm_cost_map) {
  return make_fake_runtime_only_cost_estimator(
      [op_cost_map](RuntimeOnlyOpCostEstimateKey const &k) {
        return op_cost_map.at(k);
      },
      [comm_cost_map](TensorSetMovement const &m) {
        return comm_cost_map.at(m);
      });
}

RuntimeOnlyCostEstimator make_fake_constant_runtime_only_cost_estimator(
    milliseconds_t forward_op_cost,
    milliseconds_t backward_op_cost,
    milliseconds_t comm_cost) {
  return make_fake_runtime_only_cost_estimator(
      [=](RuntimeOnlyOpCostEstimateKey const &op) {
        return RuntimeOnlyOpCostMetrics{forward_op_cost, backward_op_cost};
      },
      [=](TensorSetMovement const &op) { return comm_cost; });
}

} // namespace FlexFlow
