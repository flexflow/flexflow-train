#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ARGS_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_ARGS_BACKING_H

#include "local-execution/local_args_backing.dtg.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "pcg/computation_graph.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/task_binding.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/training_computation_graph.dtg.h"
#include "task-spec/training_layer_plus_context.dtg.h"

namespace FlexFlow {

LocalArgsBacking make_local_computation_args_backing_with_empty_device_states(
    RuntimeArgConfig const &);

std::optional<DeviceSpecificDeviceStates>
    get_per_device_op_state_if_exists(LocalArgsBacking const &,
                                      layer_guid_t const &);

std::unordered_map<slot_id_t, ConcreteArgSpec>
    construct_arg_slots_backing(TaskBinding const &, RuntimeArgConfig const &);

std::optional<DeviceSpecificDeviceStates>
    create_per_device_op_state(LocalTaskRegistry const &,
                               LocalTensorBacking const &,
                               RuntimeArgConfig const &,
                               Allocator &,
                               TrainingLayerPlusContext const &);

TaskArgumentAccessor get_task_arg_accessor(LocalTensorBacking const &,
                                           RuntimeArgConfig const &,
                                           TaskInvocation const &,
                                           Allocator &);

LocalArgsBacking make_local_args_backing_for_computation_graph(
    LocalTaskRegistry const &,
    TrainingComputationGraph const &,
    RuntimeArgConfig const &,
    LocalTensorBacking const &,
    Allocator &);

} // namespace FlexFlow

#endif
