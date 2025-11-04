#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.h"
#include "compiler/machine_mapping/machine_view.h"

namespace FlexFlow {

MachineSpaceCoordinate
  machine_space_stencil_compute_machine_coord(
    MachineSpaceStencil const &machine_space_stencil,
    TaskSpaceCoordinate const &task_space_coordinate) {

  return get_machine_space_coordinate(
    /*operator_task_space=*/machine_space_stencil.operator_task_space,
    /*machine_view=*/machine_space_stencil.machine_view,
    /*task_space_coordinate=*/task_space_coordinate);
}


} // namespace FlexFlow
