#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/device_specific_device_states.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/training_layer_plus_context.dtg.h"

namespace FlexFlow {

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    TrainingLayerPlusContext const &training_layer,
    std::optional<DeviceSpecificDeviceStates> const &device_specific_device_states);

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    layer_guid_t const &layer_guid,
    std::vector<forward_tensor_guid_t> const &input_tensors,
    std::vector<gradient_tensor_guid_t> const &input_gradient_tensors,
    std::vector<TensorShape> const &input_tensor_shapes,
    std::vector<forward_tensor_guid_t> const &weight_tensors,
    std::vector<gradient_tensor_guid_t> const &weight_gradient_tensors,
    std::vector<forward_tensor_guid_t> const &output_tensors,
    std::vector<gradient_tensor_guid_t> const &output_gradient_tensors,
    std::optional<DeviceSpecificDeviceStates> const &device_specific_device_states);

ConcreteArgSpec lower_to_concrete_arg_spec(RuntimeArgRefSpec const &,
                                           RuntimeArgConfig const &);

ConcreteArgSpec lower_to_concrete_arg_spec(
    OpArgRefSpec const &,
    std::vector<TensorShape> const &,
    layer_guid_t const &,
    std::optional<DeviceSpecificDeviceStates> const &);

} // namespace FlexFlow

#endif
