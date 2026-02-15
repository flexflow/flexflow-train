#include "realm-execution/tasks/impl/device_handle_init_task.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

struct DeviceHandleInitReturnTaskArgs {
public:
  DeviceHandleInitReturnTaskArgs() = delete;
  DeviceHandleInitReturnTaskArgs(
      DeviceSpecificManagedPerDeviceFFHandle result,
      Realm::Processor origin_proc,
      DeviceSpecificManagedPerDeviceFFHandle *origin_result_ptr)
      : result(result), origin_proc(origin_proc),
        origin_result_ptr(origin_result_ptr) {}

public:
  DeviceSpecificManagedPerDeviceFFHandle result;
  Realm::Processor origin_proc;
  DeviceSpecificManagedPerDeviceFFHandle *origin_result_ptr;
};

void device_handle_init_return_task_body(void const *args,
                                         size_t arglen,
                                         void const *userdata,
                                         size_t userlen,
                                         Realm::Processor proc) {
  ASSERT(arglen == sizeof(DeviceHandleInitReturnTaskArgs));
  DeviceHandleInitReturnTaskArgs task_args =
      *reinterpret_cast<DeviceHandleInitReturnTaskArgs const *>(args);

  ASSERT(task_args.origin_proc.address_space() == proc.address_space());
  *task_args.origin_result_ptr = task_args.result;
}

Realm::Event spawn_device_handle_init_return_task(
    RealmContext &ctx,
    Realm::Processor origin_proc,
    DeviceSpecificManagedPerDeviceFFHandle const &result,
    DeviceSpecificManagedPerDeviceFFHandle *origin_result_ptr,
    Realm::Event precondition) {
  DeviceHandleInitReturnTaskArgs task_args{
      result, origin_proc, origin_result_ptr};

  return ctx.spawn_task(origin_proc,
                        task_id_t::DEVICE_HANDLE_INIT_RETURN_TASK_ID,
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
