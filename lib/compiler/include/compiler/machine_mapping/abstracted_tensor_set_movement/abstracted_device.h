#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_DEVICE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_ABSTRACTED_DEVICE_H

#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_device.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

MachineSpaceCoordinate concretize_abstracted_device(
    AbstractedDevice const &abstracted_device,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const
        &machine_space_stencils);

} // namespace FlexFlow

#endif
