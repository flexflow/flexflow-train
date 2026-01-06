#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOWER_OP_TASK_INVOCATION_TO_RUNTIME_TASK_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOWER_OP_TASK_INVOCATION_TO_RUNTIME_TASK_INVOCATION_H

#include "kernels/device_handle_t.dtg.h"
#include "task-spec/concrete_arg_spec.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/ops/op_arg_ref_spec.h"

namespace FlexFlow {

// TODO(@lockshaw)(#pr): 
// ConcreteArgSpec
//   lower_runtime_arg_ref_spec_to_concrete_arg_spec(
//     RuntimeArgRefSpec const &,
//     RuntimeArgConfig const &r,
//     DeviceSpecific<device_handle_t> const &,
//     std::function<std::optional<DeviceSpecificPerDeviceOpState>(symbolic_layer_guid_t)> const &);

// ConcreteArgSpec lower_argumentless_arg_ref_to_concrete_arg_spec(
//     ArgumentlessRuntimeArgRefType,
//     RuntimeArgConfig const &,
//     DeviceSpecific<device_handle_t> const &);


} // namespace FlexFlow

#endif
