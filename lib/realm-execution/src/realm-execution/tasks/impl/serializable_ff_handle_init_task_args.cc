#include "realm-execution/tasks/impl/serializable_ff_handle_init_task_args.h"

namespace FlexFlow {

SerializableFfHandleInitTaskArgs
    ff_handle_init_task_args_to_serializable(FfHandleInitTaskArgs const &args) {
  return SerializableFfHandleInitTaskArgs{
      /*workSpaceSize=*/args.workSpaceSize,
      /*allowTensorOpMathConversion=*/args.allowTensorOpMathConversion,
      /*origin_proc=*/realm_processor_to_serializable(args.origin_proc),
      /*origin_result_ptr=*/reinterpret_cast<uintptr_t>(args.origin_result_ptr),
  };
}

FfHandleInitTaskArgs ff_handle_init_task_args_from_serializable(
    SerializableFfHandleInitTaskArgs const &args) {
  return FfHandleInitTaskArgs{
      /*workSpaceSize=*/args.workSpaceSize,
      /*allowTensorOpMathConversion=*/args.allowTensorOpMathConversion,
      /*origin_proc=*/realm_processor_from_serializable(args.origin_proc),
      /*origin_result_ptr=*/
      reinterpret_cast<DeviceSpecificManagedPerDeviceFFHandle *>(
          args.origin_result_ptr),
  };
}

} // namespace FlexFlow
