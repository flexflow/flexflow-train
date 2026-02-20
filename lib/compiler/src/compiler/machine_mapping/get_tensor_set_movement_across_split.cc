#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/sum.h"
#include "utils/containers/values.h"

namespace FlexFlow {

TensorSetMovement get_tensor_set_movement_across_split(
    TransitiveReducedPCG const &tr_pcg,
    PCGBinarySeriesSplit const &split,
    ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
    ParallelLayerGuidObliviousMachineMapping const &post_mapping) {

  AbstractedTensorSetMovement abstracted =
      get_abstracted_tensor_set_movement_across_split(tr_pcg, split);

  auto get_task_spaces = [&](PCGBinarySPDecomposition const &t)
      -> std::unordered_map<BinaryTreePath, OperatorTaskSpace> {
    return map_values(pcg_sp_tree_get_path_to_leaf_map(t),
                      [&](parallel_layer_guid_t parallel_layer_guid) {
                        return get_operator_task_space(tr_pcg.full_pcg,
                                                       parallel_layer_guid);
                      });
  };

  std::unordered_map<BinaryTreePath, MachineSpaceStencil> pre_stencils =
      get_machine_stencils_for_decomposition(
          tr_pcg.full_pcg, split.get_left_child(), pre_mapping);

  std::unordered_map<BinaryTreePath, MachineSpaceStencil> post_stencils =
      get_machine_stencils_for_decomposition(
          tr_pcg.full_pcg, split.get_right_child(), post_mapping);

  return concretize_abstracted_tensor_set_movement(
      abstracted,
      /*pre_machine_stencils=*/pre_stencils,
      /*post_machine_stencils=*/post_stencils);
}

} // namespace FlexFlow
