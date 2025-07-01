#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

UnmappedOpCostEstimateKey get_unmapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg, 
    OptimizerAttrs const &optimizer_attrs,
    parallel_layer_guid_t const &layer) {
  return unmapped_op_cost_estimate_key_from_runtime_only(
    get_unmapped_runtime_only_op_cost_estimate_key_for_layer(pcg, layer),
    optimizer_attrs);
}

UnmappedOpCostEstimateKey 
  unmapped_op_cost_estimate_key_from_runtime_only(
    UnmappedRuntimeOnlyOpCostEstimateKey const &runtime_only,
    OptimizerAttrs const &optimizer_attrs) {
  return UnmappedOpCostEstimateKey{
    /*op_attrs=*/runtime_only.op_attrs,
    /*input_shapes=*/runtime_only.input_shapes,
    /*weight_shapes=*/runtime_only.weight_shapes,
    /*output_shapes=*/runtime_only.output_shapes,
    /*optimizer_attrs=*/optimizer_attrs,
  };
}

UnmappedRuntimeOnlyOpCostEstimateKey
  runtime_only_from_unmapped_op_cost_estimate_key(
    UnmappedOpCostEstimateKey const &unmapped_op_cost_estimate_key) {
  return UnmappedRuntimeOnlyOpCostEstimateKey{
    /*op_attrs=*/unmapped_op_cost_estimate_key.op_attrs,
    /*input_shapes=*/unmapped_op_cost_estimate_key.input_shapes,
    /*weight_shapes=*/unmapped_op_cost_estimate_key.weight_shapes,
    /*output_shapes=*/unmapped_op_cost_estimate_key.output_shapes,
  };
}

OpCostEstimateKey
    map_unmapped_op_cost_estimate_key(UnmappedOpCostEstimateKey const &unmapped,
                                      MachineView const &machine_view) {
  return OpCostEstimateKey{
      /*op_attrs=*/unmapped.op_attrs,
      /*input_shapes=*/unmapped.input_shapes,
      /*weight_shapes=*/unmapped.weight_shapes,
      /*output_shapes=*/unmapped.output_shapes,
      /*optimizer_attrs=*/unmapped.optimizer_attrs,
      /*machine_view=*/machine_view,
  };
}

} // namespace FlexFlow
