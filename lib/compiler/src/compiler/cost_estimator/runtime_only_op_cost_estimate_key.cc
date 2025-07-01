#include "compiler/cost_estimator/runtime_only_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"

namespace FlexFlow {

RuntimeOnlyOpCostEstimateKey get_mapped_runtime_only_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg,
    parallel_layer_guid_t const &parallel_layer_guid,
    MachineView const &machine_view) {
  return map_unmapped_runtime_only_op_cost_estimate_key(
      get_unmapped_runtime_only_op_cost_estimate_key_for_layer(pcg, parallel_layer_guid), 
      machine_view);
}


} // namespace FlexFlow
