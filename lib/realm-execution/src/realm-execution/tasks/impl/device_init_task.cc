#include "realm-execution/tasks/impl/device_init_task.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/optional.h"
#include <type_traits>

namespace FlexFlow {

// TODO: at some point we're going to have to actually serialize these, but for
// now just pass the pointer and assume we're running inside a single address
// space
struct DeviceInitTaskArgs {
public:
  DynamicNodeInvocation const *invocation;
  Realm::Processor origin_proc;
  DeviceSpecificPerDeviceOpState *origin_result_ptr;
};
static_assert(std::has_unique_object_representations_v<DeviceInitTaskArgs>);

void device_init_task_body(void const *args,
                           size_t arglen,
                           void const *userdata,
                           size_t userlen,
                           Realm::Processor proc) {
  ASSERT(arglen == sizeof(DeviceInitTaskArgs));
  DeviceInitTaskArgs task_args =
      *reinterpret_cast<DeviceInitTaskArgs const *>(args);

  // FIXME: not safe to dereference unless we're on the same address space
  ASSERT(task_args.origin_proc.address_space() == proc.address_space());

  RealmContext ctx{proc};
  NOT_IMPLEMENTED();
}

Realm::Event
    spawn_device_init_task(RealmContext &ctx,
                           Realm::Processor &target_proc,
                           DynamicNodeInvocation const &invocation,
                           std::optional<OptimizerAttrs> const &optimizer_attrs,
                           DeviceSpecificPerDeviceOpState *result_ptr) {
  DeviceInitTaskArgs task_args;
  task_args.invocation = &invocation;
  task_args.origin_proc = ctx.get_current_processor();
  task_args.origin_result_ptr = result_ptr;

  return ctx.spawn_task(target_proc,
                        assert_unwrap(get_init_task_id_for_op_attrs(
                            assert_unwrap(invocation.node_attrs.op_attrs))),
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{});
}

} // namespace FlexFlow
