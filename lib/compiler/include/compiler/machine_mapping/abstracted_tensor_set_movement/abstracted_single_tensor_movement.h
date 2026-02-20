#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_SINGLE_TENSOR_MOVEMENT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_SINGLE_TENSOR_MOVEMENT_H

#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_communication.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"

namespace FlexFlow {

std::unordered_set<BinaryTreePath>
    abstracted_single_tensor_movement_get_dst_layers(
        AbstractedSingleTensorMovement const &);

AbstractedSingleTensorMovement merge_abstracted_single_tensor_movements(
    std::unordered_multiset<AbstractedSingleTensorMovement> const &);

AbstractedSingleTensorMovement
    abstracted_single_tensor_movement_from_communications(
        BinaryTreePath const &src_op_tree_path,
        std::unordered_set<AbstractedSingleTensorCommunication> const
            &communications);

TensorSetMovement concretize_abstracted_single_tensor_movement(
    AbstractedSingleTensorMovement const &,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const
        &pre_machine_stencils,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const
        &post_machine_stencils);

} // namespace FlexFlow

#endif
