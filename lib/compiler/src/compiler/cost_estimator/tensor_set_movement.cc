#include "compiler/cost_estimator/tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/full_binary_tree/binary_tree_path.dtg.h"

namespace FlexFlow {

TensorSetMovement 
  get_tensor_set_movement_from_pcg_edge(
    ParallelComputationGraphEdge const &edge,
    ParallelComputationGraph const &pcg,
    MachineView const &src_mv,
    MachineView const &dst_mv) {

  parallel_layer_guid_t src = get_src_layer(edge);
  parallel_layer_guid_t dst = get_dst_layer(edge);

  BinaryTreePath src_path = BinaryTreePath{{BinaryTreePathEntry::LEFT_CHILD}};
  BinaryTreePath dst_path = BinaryTreePath{{BinaryTreePathEntry::RIGHT_CHILD}};

  std::unordered_set<AbstractedSingleCommunication> abstracted_single_communications = 
    get_abstracted_single_communications_along_edge(
      /*pcg=*/pcg, 
      /*edge=*/edge, 
      /*src_path=*/src_path,
      /*dst_path=*/dst_path);

  AbstractedTensorSetMovement abstracted_tensor_set_movement 
    = abstracted_tensor_set_movement_from_single_communications(unordered_multiset_of(abstracted_single_communications));

  MachineSpaceStencil src_machine_stencil = MachineSpaceStencil{
    /*operator_task_space=*/get_operator_task_space(pcg, src),
    /*machine_view=*/src_mv,
  };

  MachineSpaceStencil dst_machine_stencil = MachineSpaceStencil{
    /*operator_task_space=*/get_operator_task_space(pcg, dst),
    /*machine_view=*/dst_mv,
  };

  return concretize_abstracted_tensor_set_movement(
    abstracted_tensor_set_movement,
    /*pre_machine_stencils=*/std::unordered_map<BinaryTreePath, MachineSpaceStencil>{
      {src_path, src_machine_stencil},
    },
    /*post_machine_stencils=*/std::unordered_map<BinaryTreePath, MachineSpaceStencil>{
      {dst_path, dst_machine_stencil},
    });
}

} // namespace FlexFlow
