#include "compiler/cost_estimator/runtime_only_cost_estimator.h"

namespace FlexFlow {

RuntimeOnlyCostEstimator::RuntimeOnlyCostEstimator(std::shared_ptr<IRuntimeOnlyCostEstimator> implementation_ptr)
    : implementation_ptr(implementation_ptr) {}

RuntimeOnlyOpCostMetrics RuntimeOnlyCostEstimator::estimate_cost(RuntimeOnlyOpCostEstimateKey const &k) const {
  return this->implementation_ptr->estimate_cost(k);
}

milliseconds_t RuntimeOnlyCostEstimator::estimate_cost(TensorSetMovement const &m) const {
  return this->implementation_ptr->estimate_cost(m);
}

} // namespace FlexFlow
