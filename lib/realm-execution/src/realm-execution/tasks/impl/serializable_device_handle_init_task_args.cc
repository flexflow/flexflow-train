#include "realm-execution/tasks/impl/serializable_device_handle_init_task_args.h"

namespace FlexFlow {

SerializableDeviceHandleInitTaskArgs
    device_handle_init_task_args_to_serializable(
        DeviceHandleInitTaskArgs const &args) {
  return SerializableDeviceHandleInitTaskArgs{
      /*workSpaceSize=*/args.workSpaceSize,
      /*allowTensorOpMathConversion=*/args.allowTensorOpMathConversion,
      /*origin_proc=*/realm_processor_to_serializable(args.origin_proc),
      /*origin_result_ptr=*/reinterpret_cast<uintptr_t>(args.origin_result_ptr),
  };
}

DeviceHandleInitTaskArgs device_handle_init_task_args_from_serializable(
    SerializableDeviceHandleInitTaskArgs const &args) {
  return DeviceHandleInitTaskArgs{
      /*workSpaceSize=*/args.workSpaceSize,
      /*allowTensorOpMathConversion=*/args.allowTensorOpMathConversion,
      /*origin_proc=*/realm_processor_from_serializable(args.origin_proc),
      /*origin_result_ptr=*/
      reinterpret_cast<DeviceSpecificManagedPerDeviceFFHandle *>(
          args.origin_result_ptr),
  };
}

} // namespace FlexFlow
