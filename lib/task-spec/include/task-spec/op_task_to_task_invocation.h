#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H

#include "pcg/cg_operator_tensor_shape_signature.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/device_specific_device_states.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/training_layer_plus_context.dtg.h"
#include "task-spec/training_layer_tensor_group_signature.dtg.h"
#include "task-spec/training_parallel_layer_plus_context.dtg.h"

namespace FlexFlow {

TaskInvocation
    lower_to_task_invocation(OpTaskInvocation const &op_task_invocation,
                             TrainingLayerPlusContext const &training_layer,
                             std::optional<DeviceSpecificDeviceStates> const
                                 &device_specific_device_states);

TaskInvocation
    pcg_lower_to_task_invocation(OpTaskInvocation const &op_task_invocation,
                                 TrainingParallelLayerPlusContext const &training_parallel_layer,
                                 std::optional<DeviceSpecificDeviceStates> const 
                                    &device_specific_device_states);


std::pair<tensor_sub_slot_id_t, training_tensor_guid_t> lower_tensor_binding(
    TrainingLayerTensorGroupSignature const &training_layer_signature,
    SlotGradId const &slot_grad_id,
    OpTensorSpec const &op_tensor_spec);

TaskArgSpec lower_to_task_arg_spec(
    OpArgSpec const &op_arg_spec,
    CGOperatorTensorShapeSignature const &op_shape_signature,
    layer_guid_t const &layer_guid,
    std::optional<DeviceSpecificDeviceStates> const
        &device_specific_device_states);

ConcreteArgSpec lower_to_concrete_arg_spec(RuntimeArgRefSpec const &,
                                           RuntimeArgConfig const &);

ConcreteArgSpec lower_to_concrete_arg_spec(
    OpArgRefSpec const &,
    CGOperatorTensorShapeSignature const &,
    layer_guid_t const &,
    std::optional<DeviceSpecificDeviceStates> const &);

} // namespace FlexFlow

#endif
