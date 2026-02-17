#include "realm-execution/tasks/impl/serializable_device_state_init_task_args.h"
#include "realm-execution/tasks/serializer/serializable_realm_instance.h"
#include "realm-execution/tasks/serializer/serializable_realm_processor.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/containers/map_keys_and_values.h"

namespace FlexFlow {

SerializableDeviceStateInitTaskArgs device_state_init_task_args_to_serializable(
    DeviceStateInitTaskArgs const &args) {
  return SerializableDeviceStateInitTaskArgs{
      /*invocation=*/dynamic_node_invocation_to_serializable(args.invocation),
      /*tensor_backing*/
      map_keys_and_values(args.tensor_backing,
                          dynamic_value_attrs_to_serializable,
                          realm_instance_to_serializable),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/args.device_handle.serialize(),
      /*iteration_config=*/args.iteration_config,
      /*optimizer_attrs=*/args.optimizer_attrs,
      /*origin_proc=*/realm_processor_to_serializable(args.origin_proc),
      /*origin_result_ptr=*/reinterpret_cast<uintptr_t>(args.origin_result_ptr),
  };
}

DeviceStateInitTaskArgs device_state_init_task_args_from_serializable(
    SerializableDeviceStateInitTaskArgs const &args) {
  return DeviceStateInitTaskArgs{
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
      /*origin_proc=*/realm_processor_from_serializable(args.origin_proc),
      /*origin_result_ptr=*/
      reinterpret_cast<DeviceSpecificPerDeviceOpState *>(
          args.origin_result_ptr),
  };
}

} // namespace FlexFlow
