#include "compiler/machine_mapping/memory_optimization/pareto_optimal_machine_mapping.h"
#include "compiler/cost_estimator/op_cost_metrics.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

bool is_pareto_optimal_in(
    ParetoOptimalMachineMapping const &m,
    std::unordered_set<ParetoOptimalMachineMapping> const &others) {
  return is_pareto_optimal_in(
      m.cost, transform(others, [](ParetoOptimalMachineMapping const &m) {
        return m.cost;
      }));
}

} // namespace FlexFlow
