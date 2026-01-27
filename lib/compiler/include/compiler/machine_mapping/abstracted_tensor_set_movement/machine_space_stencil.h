#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_MACHINE_SPACE_STENCIL_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ABSTRACTED_TENSOR_SET_MOVEMENT_MACHINE_SPACE_STENCIL_H

#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

MachineSpaceCoordinate
    machine_space_stencil_compute_machine_coord(MachineSpaceStencil const &,
                                                TaskSpaceCoordinate const &);

} // namespace FlexFlow

#endif
