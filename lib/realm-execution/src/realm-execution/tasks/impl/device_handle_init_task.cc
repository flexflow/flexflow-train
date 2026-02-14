#include "realm-execution/tasks/impl/device_handle_init_task.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/tasks/impl/device_handle_init_return_task.h"
#include "realm-execution/tasks/impl/device_handle_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_device_handle_init_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include <type_traits>

namespace FlexFlow {

static std::optional<ManagedPerDeviceFFHandle *>
    make_device_handle_for_processor(Realm::Processor processor,
                                     size_t workSpaceSize,
                                     bool allowTensorOpMathConversion) {
  switch (processor.kind()) {
    case Realm::Processor::LOC_PROC:
      return std::nullopt;
    case Realm::Processor::TOC_PROC:
      return new ManagedPerDeviceFFHandle{initialize_multi_gpu_handle(
          /*num_ranks=*/Realm::Machine::get_machine().get_address_space_count(),
          /*my_rank=*/processor.address_space(),
          /*workSpaceSize=*/workSpaceSize,
          /*allowTensorOpMathConversion=*/allowTensorOpMathConversion)};
    default:
      PANIC("Unhandled Realm::ProcessorKind",
            fmt::to_string(int{processor.kind()}));
  }
}

void device_handle_init_task_body(void const *args,
                                  size_t arglen,
                                  void const *userdata,
                                  size_t userlen,
                                  Realm::Processor proc) {
  DeviceHandleInitTaskArgs task_args =
      device_handle_init_task_args_from_serializable(
          deserialize_task_args<SerializableDeviceHandleInitTaskArgs>(args,
                                                                      arglen));

  RealmContext ctx{proc};
  DeviceSpecificManagedPerDeviceFFHandle managed_handle =
      make_device_specific_managed_handle(
          ctx.get_current_device_idx(),
          make_device_handle_for_processor(
              proc,
              task_args.workSpaceSize,
              task_args.allowTensorOpMathConversion));

  spawn_device_handle_init_return_task(ctx,
                                       task_args.origin_proc,
                                       managed_handle,
                                       task_args.origin_result_ptr,
                                       Realm::Event::NO_EVENT);
}

Realm::Event spawn_device_handle_init_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    size_t workSpaceSize,
    bool allowTensorOpMathConversion,
    DeviceSpecificManagedPerDeviceFFHandle *result_ptr,
    Realm::Event precondition) {
  DeviceHandleInitTaskArgs task_args{
      workSpaceSize,
      allowTensorOpMathConversion,
      ctx.get_current_processor(),
      result_ptr,
  };

  std::string args = serialize_task_args(
      device_handle_init_task_args_to_serializable(task_args));
  return ctx.spawn_task(target_proc,
                        task_id_t::DEVICE_HANDLE_INIT_TASK_ID,
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
