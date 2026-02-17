#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/serializer/serializable_realm_instance.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/containers/map_keys_and_values.h"

namespace FlexFlow {

SerializableOpTaskArgs op_task_args_to_serializable(OpTaskArgs const &args) {
  return SerializableOpTaskArgs{
      /*invocation=*/dynamic_node_invocation_to_serializable(args.invocation),
      /*tensor_backing*/
      map_keys_and_values(args.tensor_backing,
                          dynamic_value_attrs_to_serializable,
                          realm_instance_to_serializable),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/args.device_handle.serialize(),
      /*iteration_config=*/args.iteration_config,
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

OpTaskArgs op_task_args_from_serializable(SerializableOpTaskArgs const &args) {
  return OpTaskArgs{
      /*invocation=*/dynamic_node_invocation_from_serializable(args.invocation),
      /*tensor_backing*/
      map_keys_and_values(args.tensor_backing,
                          dynamic_value_attrs_from_serializable,
                          realm_instance_from_serializable),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/
      DeviceSpecificManagedPerDeviceFFHandle::deserialize(args.device_handle),
      /*iteration_config=*/args.iteration_config,
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

} // namespace FlexFlow
