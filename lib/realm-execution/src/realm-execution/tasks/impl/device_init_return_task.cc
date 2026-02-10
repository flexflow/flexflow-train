#include "realm-execution/tasks/impl/device_init_task.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

// FIXME: Can't make this trivially copyable?
struct DeviceInitReturnTaskArgs {
public:
  DeviceInitReturnTaskArgs() = delete;
  DeviceInitReturnTaskArgs(DeviceSpecificPerDeviceOpState result,
                           Realm::Processor origin_proc,
                           DeviceSpecificPerDeviceOpState *origin_result_ptr)
      : result(result), origin_proc(origin_proc),
        origin_result_ptr(origin_result_ptr) {}

public:
  DeviceSpecificPerDeviceOpState result;
  Realm::Processor origin_proc;
  DeviceSpecificPerDeviceOpState *origin_result_ptr;
};

void device_init_return_task_body(void const *args,
                                  size_t arglen,
                                  void const *userdata,
                                  size_t userlen,
                                  Realm::Processor proc) {
  ASSERT(arglen == sizeof(DeviceInitReturnTaskArgs));
  DeviceInitReturnTaskArgs task_args =
      *reinterpret_cast<DeviceInitReturnTaskArgs const *>(args);

  ASSERT(task_args.origin_proc.address_space() == proc.address_space());
  *task_args.origin_result_ptr = task_args.result;
}

Realm::Event spawn_device_init_return_task(
    RealmContext &ctx,
    Realm::Processor origin_proc,
    DeviceSpecificPerDeviceOpState const &result,
    DeviceSpecificPerDeviceOpState *origin_result_ptr) {
  DeviceInitReturnTaskArgs task_args{result, origin_proc, origin_result_ptr};

  return ctx.spawn_task(origin_proc,
                        task_id_t::DEVICE_INIT_RETURN_TASK_ID,
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{});
}

} // namespace FlexFlow
