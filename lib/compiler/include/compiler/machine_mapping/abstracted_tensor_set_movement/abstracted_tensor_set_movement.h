#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_H

#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_communication.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

AbstractedTensorSetMovement empty_abstracted_tensor_set_movement();

AbstractedTensorSetMovement 
  abstracted_tensor_set_movement_from_single_tensor_movement(
    AbstractedSingleTensorMovement const &);

std::unordered_set<BinaryTreePath>
    get_src_layers(AbstractedTensorSetMovement const &);
std::unordered_set<BinaryTreePath>
    get_dst_layers(AbstractedTensorSetMovement const &);

TensorSetMovement concretize_abstracted_tensor_set_movement(
    AbstractedTensorSetMovement const &,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &pre_machine_stencils,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &post_machine_stencils);

} // namespace FlexFlow

#endif
