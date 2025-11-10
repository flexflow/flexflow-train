#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_DEVICE_STATES_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_DEVICE_STATES_BACKING_H

#include "local-execution/local_device_states_backing.dtg.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "pcg/computation_graph.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature_with_shapes.dtg.h"

namespace FlexFlow {

LocalDeviceStatesBacking make_local_device_states_backing_for_computation_graph(
    LocalTaskRegistry const &,
    std::unordered_map<layer_guid_t, SymbolicLayerTrainingTensorGroupSignatureWithShapes> const &,
    RuntimeArgConfig const &runtime_arg_config,
    LocalTensorBacking const &,
    Allocator &);

std::optional<DeviceSpecificPerDeviceOpState>
    get_per_device_op_state_if_exists(LocalDeviceStatesBacking const &,
                                      layer_guid_t const &);

} // namespace FlexFlow

#endif
