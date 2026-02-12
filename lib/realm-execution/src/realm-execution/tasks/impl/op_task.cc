#include "realm-execution/tasks/impl/op_task.h"
#include "local-execution/task_execution.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/per_device_op_state.h"
#include "utils/optional.h"
#include <type_traits>

namespace FlexFlow {

// TODO: at some point we're going to have to actually serialize these, but for
// now just pass the pointer and assume we're running inside a single address
// space
struct OpTaskArgs {
public:
  OpTaskArgs() = delete;
  OpTaskArgs(DynamicNodeInvocation const *invocation,
             ProfilingSettings const *profiling_settings,
             FFIterationConfig const *iteration_config,
             std::optional<OptimizerAttrs> const *optimizer_attrs,
             Realm::Processor origin_proc)
      : invocation(invocation), profiling_settings(profiling_settings),
        iteration_config(iteration_config), optimizer_attrs(optimizer_attrs) {}

public:
  DynamicNodeInvocation const *invocation;
  ProfilingSettings const *profiling_settings;
  FFIterationConfig const *iteration_config;
  std::optional<OptimizerAttrs> const *optimizer_attrs;
  Realm::Processor origin_proc;
};
static_assert(std::has_unique_object_representations_v<OpTaskArgs>);

void op_task_body(void const *args,
                  size_t arglen,
                  void const *userdata,
                  size_t userlen,
                  Realm::Processor proc) {
  ASSERT(arglen == sizeof(OpTaskArgs));
  OpTaskArgs task_args = *reinterpret_cast<OpTaskArgs const *>(args);

  // FIXME: serialize instead of passing pointers around
  ASSERT(task_args.origin_proc.address_space() == proc.address_space());

  RealmContext ctx{proc};
  execute_dynamic_node_invocation(
      /*invocation=*/*task_args.invocation,
      /*allocator=*/ctx.get_current_device_allocator(),
      /*profiling_settings=*/*task_args.profiling_settings,
      /*ff_handle=*/ctx.get_current_device_handle(),
      /*per_device_op_state=*/
      transform(task_args.invocation->node_attrs.per_device_op_state,
                [&](DeviceSpecificPerDeviceOpState const &op_state) {
                  return get_device_state_from_device_specific(
                      op_state, ctx.get_current_device_idx());
                }),
      /*iteration_config=*/*task_args.iteration_config,
      /*optimizer_attrs=*/*task_args.optimizer_attrs,
      /*device_idx=*/ctx.get_current_device_idx());
}

Realm::Event
    spawn_op_task(RealmContext &ctx,
                  Realm::Processor target_proc,
                  DynamicNodeInvocation const &invocation,
                  ProfilingSettings const &profiling_settings,
                  FFIterationConfig const &iteration_config,
                  std::optional<OptimizerAttrs> const &optimizer_attrs) {
  OpTaskArgs task_args{&invocation,
                       &profiling_settings,
                       &iteration_config,
                       &optimizer_attrs,
                       ctx.get_current_processor()};
  return ctx.spawn_task(
      target_proc,
      assert_unwrap(get_task_id_for_op(invocation.node_attrs, optimizer_attrs)),
      &task_args,
      sizeof(task_args),
      Realm::ProfilingRequestSet{});
}

} // namespace FlexFlow
