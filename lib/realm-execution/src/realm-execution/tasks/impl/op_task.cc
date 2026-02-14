#include "realm-execution/tasks/impl/op_task.h"
#include "local-execution/task_execution.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/per_device_op_state.h"
#include "utils/optional.h"
#include <type_traits>

namespace FlexFlow {

void op_task_body(void const *args,
                  size_t arglen,
                  void const *userdata,
                  size_t userlen,
                  Realm::Processor proc) {
  OpTaskArgs task_args = op_task_args_from_serializable(
      deserialize_task_args<SerializableOpTaskArgs>(args, arglen));

  RealmContext ctx{proc};
  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_handle(
          task_args.device_handle, ctx.get_current_device_idx());
  execute_dynamic_node_invocation(
      /*invocation=*/task_args.invocation,
      /*allocator=*/ctx.get_current_device_allocator(),
      /*profiling_settings=*/task_args.profiling_settings,
      /*ff_handle=*/device_handle,
      /*per_device_op_state=*/
      transform(task_args.invocation.node_attrs.per_device_op_state,
                [&](DeviceSpecificPerDeviceOpState const &op_state) {
                  return get_device_state_from_device_specific(
                      op_state, ctx.get_current_device_idx());
                }),
      /*iteration_config=*/task_args.iteration_config,
      /*optimizer_attrs=*/task_args.optimizer_attrs,
      /*device_idx=*/ctx.get_current_device_idx());
}

Realm::Event
    spawn_op_task(RealmContext &ctx,
                  Realm::Processor target_proc,
                  DynamicNodeInvocation const &invocation,
                  ProfilingSettings const &profiling_settings,
                  DeviceSpecificManagedPerDeviceFFHandle const &device_handle,
                  FFIterationConfig const &iteration_config,
                  std::optional<OptimizerAttrs> const &optimizer_attrs,
                  Realm::Event precondition) {
  OpTaskArgs task_args{invocation,
                       profiling_settings,
                       device_handle,
                       iteration_config,
                       optimizer_attrs};
  std::string args =
      serialize_task_args(op_task_args_to_serializable(task_args));
  return ctx.spawn_task(
      target_proc,
      assert_unwrap(get_task_id_for_op(invocation.node_attrs, optimizer_attrs)),
      args.data(),
      args.size(),
      Realm::ProfilingRequestSet{},
      precondition);
}

} // namespace FlexFlow
