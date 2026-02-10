#include "realm-execution/tasks/impl/device_init_task.h"
#include "local-execution/device_state_initialization.h"
#include "realm-execution/tasks/impl/device_init_return_task.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/optional.h"
#include <optional>
#include <type_traits>

namespace FlexFlow {

// TODO: at some point we're going to have to actually serialize these, but for
// now just pass the pointer and assume we're running inside a single address
// space
struct DeviceInitTaskArgs {
  DeviceInitTaskArgs() = delete;
  DeviceInitTaskArgs(DynamicNodeInvocation const *invocation,
                     ProfilingSettings const *profiling_settings,
                     FFIterationConfig const *iteration_config,
                     OptimizerAttrs const *optimizer_attrs,
                     Realm::Processor origin_proc,
                     DeviceSpecificPerDeviceOpState *origin_result_ptr)
      : invocation(invocation), profiling_settings(profiling_settings),
        iteration_config(iteration_config), optimizer_attrs(optimizer_attrs),
        origin_proc(origin_proc), origin_result_ptr(origin_result_ptr) {}

public:
  DynamicNodeInvocation const *invocation;
  ProfilingSettings const *profiling_settings;
  FFIterationConfig const *iteration_config;
  OptimizerAttrs const *optimizer_attrs;
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
  DynamicNodeInvocation result_invocation =
      initialize_node(*task_args.invocation,
                      ctx.get_current_device_allocator(),
                      *task_args.profiling_settings,
                      ctx.get_current_device_handle(),
                      *task_args.iteration_config,
                      *task_args.optimizer_attrs,
                      ctx.get_current_device_idx());
  std::optional<DeviceSpecificPerDeviceOpState> result_state =
      result_invocation.node_attrs.per_device_op_state;
  if (result_state) {
    spawn_device_init_return_task(ctx,
                                  task_args.origin_proc,
                                  assert_unwrap(result_state),
                                  task_args.origin_result_ptr);
  }
}

Realm::Event
    spawn_device_init_task(RealmContext &ctx,
                           Realm::Processor &target_proc,
                           DynamicNodeInvocation const &invocation,
                           ProfilingSettings const &profiling_settings,
                           FFIterationConfig const &iteration_config,
                           OptimizerAttrs const &optimizer_attrs,
                           DeviceSpecificPerDeviceOpState *result_ptr) {
  DeviceInitTaskArgs task_args{
      &invocation,
      &profiling_settings,
      &iteration_config,
      &optimizer_attrs,
      ctx.get_current_processor(),
      result_ptr,
  };

  return ctx.spawn_task(target_proc,
                        assert_unwrap(get_init_task_id_for_op_attrs(
                            assert_unwrap(invocation.node_attrs.op_attrs))),
                        &task_args,
                        sizeof(task_args),
                        Realm::ProfilingRequestSet{});
}

} // namespace FlexFlow
