#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ALLOWED_MACHINE_VIEWS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_ALLOWED_MACHINE_VIEWS_H

#include "pcg/machine_compute_resource_slice.dtg.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

bool is_valid_machine_view(MachineView const &mv,
                           OperatorTaskSpace const &task,
                           MachineComputeResourceSlice const &ms);

std::unordered_set<MachineView>
    get_allowed_machine_views(MachineComputeResourceSlice const &machine_spec,
                              OperatorTaskSpace const &task,
                              DeviceType device_type);

} // namespace FlexFlow

#endif
