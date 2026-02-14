#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"

namespace FlexFlow {

SerializableOpTaskArgs op_task_args_to_serializable(OpTaskArgs const &args) {
  return SerializableOpTaskArgs{
      /*invocation=*/dynamic_node_invocation_to_serializable(args.invocation),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/args.device_handle.serialize(),
      /*iteration_config=*/args.iteration_config,
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

OpTaskArgs op_task_args_from_serializable(SerializableOpTaskArgs const &args) {
  return OpTaskArgs{
      /*invocation=*/dynamic_node_invocation_from_serializable(args.invocation),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/
      DeviceSpecificManagedPerDeviceFFHandle::deserialize(args.device_handle),
      /*iteration_config=*/args.iteration_config,
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

} // namespace FlexFlow
