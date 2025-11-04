#include "compiler/cost_estimator/op_cost_estimate_key.h"
#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "compiler/machine_mapping/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include <unordered_set>

namespace FlexFlow {

OpCostEstimateKey get_mapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg,
    OptimizerAttrs const &optimizer_attrs,
    parallel_layer_guid_t const &parallel_layer_guid,
    MachineView const &machine_view) {
  return map_unmapped_op_cost_estimate_key(
      get_unmapped_op_cost_estimate_key_for_layer(
          pcg, optimizer_attrs, parallel_layer_guid),
      machine_view);
}

RuntimeOnlyOpCostEstimateKey runtime_only_from_op_cost_estimate_key(
    OpCostEstimateKey const &op_cost_estimate_key) {

  return RuntimeOnlyOpCostEstimateKey{
      /*op_attrs=*/op_cost_estimate_key.op_attrs,
      /*input_shapes=*/op_cost_estimate_key.input_shapes,
      /*weight_shapes=*/op_cost_estimate_key.weight_shapes,
      /*output_shapes=*/op_cost_estimate_key.output_shapes,
      /*machine_view=*/op_cost_estimate_key.machine_view,
  };
}

OpCostEstimateKey make_op_cost_estimate_key_from_runtime_only(
    RuntimeOnlyOpCostEstimateKey const &runtime_only,
    OptimizerAttrs const &optimizer_attrs) {

  return OpCostEstimateKey{
      /*op_attrs=*/runtime_only.op_attrs,
      /*input_shapes=*/runtime_only.input_shapes,
      /*weight_shapes=*/runtime_only.weight_shapes,
      /*output_shapes=*/runtime_only.output_shapes,
      /*optimizer_attrs=*/optimizer_attrs,
      /*machine_view=*/runtime_only.machine_view,
  };
}

} // namespace FlexFlow
