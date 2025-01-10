#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"

namespace FlexFlow {

OpCostEstimateKey get_mapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg,
    parallel_layer_guid_t const &layer,
    MachineView const &machine_view) {
  return map_unmapped_op_cost_estimate_key(
      get_unmapped_op_cost_estimate_key_for_layer(pcg, layer), machine_view);
}

} // namespace FlexFlow
