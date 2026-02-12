#include "realm-execution/tasks/impl/device_handle_init_task.h"
#include "realm-execution/tasks/impl/device_handle_init_return_task.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include <type_traits>

namespace FlexFlow {

// TODO: at some point we're going to have to actually serialize these, but for
// now just pass the pointer and assume we're running inside a single address
// space
struct DeviceHandleInitTaskArgs {
  DeviceHandleInitTaskArgs() = delete;
  DeviceHandleInitTaskArgs(
      size_t workSpaceSize,
      bool allowTensorOpMathConversion,
      Realm::Processor origin_proc,
      DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>>
          *origin_result_ptr)
      : workSpaceSize(workSpaceSize),
        allowTensorOpMathConversion(allowTensorOpMathConversion),
        origin_proc(origin_proc), origin_result_ptr(origin_result_ptr) {}

public:
  size_t workSpaceSize;
  bool allowTensorOpMathConversion;
  Realm::Processor origin_proc;
  DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>> *origin_result_ptr;
};
static_assert(std::is_trivially_copy_constructible_v<DeviceHandleInitTaskArgs>);

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
  ASSERT(arglen == sizeof(DeviceHandleInitTaskArgs));
  DeviceHandleInitTaskArgs task_args =
      *reinterpret_cast<DeviceHandleInitTaskArgs const *>(args);

  // FIXME: serialize instead of passing pointers around
  ASSERT(task_args.origin_proc.address_space() == proc.address_space());

  RealmContext ctx{proc};
  DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>> managed_handle =
      DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>>::create(
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
    DeviceSpecific<std::optional<ManagedPerDeviceFFHandle *>> *result_ptr,
    Realm::Event precondition) {
  DeviceHandleInitTaskArgs task_args{
      workSpaceSize,
      allowTensorOpMathConversion,
      ctx.get_current_processor(),
      result_ptr,
  };

  return ctx.spawn_task(target_proc,
                        task_id_t::DEVICE_HANDLE_INIT_TASK_ID,
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
