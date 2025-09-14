#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_SINGLE_TENSOR_COMMUNICATION_EDGE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_SINGLE_TENSOR_COMMUNICATION_EDGE_H

#include "compiler/cost_estimator/communication_edge.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_communication_edge.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
namespace FlexFlow {

std::optional<CommunicationEdge>
  concretize_abstracted_single_tensor_communication_edge(
    AbstractedSingleTensorCommunicationEdge const &edge,
    MachineSpaceStencil const &src_machine_stencil,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &dst_machine_stencils);


} // namespace FlexFlow

#endif
