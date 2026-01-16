#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_runtime_only_op_cost_estimate_key.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

UnmappedRuntimeOnlyOpCostEstimateKey
    get_unmapped_runtime_only_op_cost_estimate_key_for_layer(
        ParallelComputationGraph const &pcg,
        parallel_layer_guid_t const &parallel_layer_guid) {
  auto get_tensor_shape = [&](parallel_tensor_guid_t const &t) {
    return get_parallel_tensor_shape(pcg, t);
  };

  return UnmappedRuntimeOnlyOpCostEstimateKey{
      /*op_attrs=*/pcg_get_op_attrs(pcg, parallel_layer_guid),
      /*input_shapes=*/
      map_values(get_incoming_inputs(pcg, parallel_layer_guid),
                 get_tensor_shape),
      /*weight_shapes=*/
      map_values(get_incoming_weights(pcg, parallel_layer_guid),
                 get_tensor_shape),
      /*output_shapes=*/
      map_values(get_layer_outputs(pcg, parallel_layer_guid), get_tensor_shape),
  };
}

RuntimeOnlyOpCostEstimateKey map_unmapped_runtime_only_op_cost_estimate_key(
    UnmappedRuntimeOnlyOpCostEstimateKey const &unmapped,
    MachineView const &machine_view) {
  return RuntimeOnlyOpCostEstimateKey{
      /*op_attrs=*/unmapped.op_attrs,
      /*input_shapes=*/unmapped.input_shapes,
      /*weight_shapes=*/unmapped.weight_shapes,
      /*output_shapes=*/unmapped.output_shapes,
      /*machine_view=*/machine_view,
  };
}

} // namespace FlexFlow
