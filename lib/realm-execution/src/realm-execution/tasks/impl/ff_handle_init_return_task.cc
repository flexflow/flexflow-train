#include "realm-execution/tasks/impl/ff_handle_init_task.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

struct FfHandleInitReturnTaskArgs {
public:
  FfHandleInitReturnTaskArgs() = delete;
  FfHandleInitReturnTaskArgs(
      DeviceSpecificPtr<ManagedPerDeviceFFHandle> result,
      Realm::Processor origin_proc,
      DeviceSpecificPtr<ManagedPerDeviceFFHandle> *origin_result_ptr)
      : result(result), origin_proc(origin_proc),
        origin_result_ptr(origin_result_ptr) {}

public:
  DeviceSpecificPtr<ManagedPerDeviceFFHandle> result;
  Realm::Processor origin_proc;
  DeviceSpecificPtr<ManagedPerDeviceFFHandle> *origin_result_ptr;
};
static_assert(std::is_trivially_copyable_v<FfHandleInitReturnTaskArgs>);

void ff_handle_init_return_task_body(void const *args,
                                     size_t arglen,
                                     void const *userdata,
                                     size_t userlen,
                                     Realm::Processor proc) {
  ASSERT(arglen == sizeof(FfHandleInitReturnTaskArgs));
  FfHandleInitReturnTaskArgs task_args =
      *reinterpret_cast<FfHandleInitReturnTaskArgs const *>(args);

  ASSERT(task_args.origin_proc.address_space() == proc.address_space());
  *task_args.origin_result_ptr = task_args.result;
}

Realm::Event spawn_ff_handle_init_return_task(
    RealmContext &ctx,
    Realm::Processor origin_proc,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &result,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> *origin_result_ptr,
    Realm::Event precondition) {
  FfHandleInitReturnTaskArgs task_args{result, origin_proc, origin_result_ptr};

  return ctx.spawn_task(origin_proc,
                        task_id_t::DEVICE_HANDLE_INIT_RETURN_TASK_ID,
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
