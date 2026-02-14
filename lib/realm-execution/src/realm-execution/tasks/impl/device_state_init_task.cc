#include "realm-execution/tasks/impl/device_state_init_task.h"
#include "local-execution/device_state_initialization.h"
#include "realm-execution/tasks/impl/device_state_init_return_task.h"
#include "realm-execution/tasks/impl/device_state_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_device_state_init_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "utils/optional.h"
#include <optional>
#include <type_traits>

namespace FlexFlow {

void device_state_init_task_body(void const *args,
                                 size_t arglen,
                                 void const *userdata,
                                 size_t userlen,
                                 Realm::Processor proc) {
  DeviceStateInitTaskArgs task_args =
      device_state_init_task_args_from_serializable(
          deserialize_task_args<SerializableDeviceStateInitTaskArgs>(args,
                                                                     arglen));

  RealmContext ctx{proc};
  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_handle(
          task_args.device_handle, ctx.get_current_device_idx());
  DynamicNodeInvocation result_invocation =
      initialize_node(task_args.invocation,
                      ctx.get_current_device_allocator(),
                      task_args.profiling_settings,
                      device_handle,
                      task_args.iteration_config,
                      task_args.optimizer_attrs,
                      ctx.get_current_device_idx());
  DeviceSpecificPerDeviceOpState result_state =
      assert_unwrap(result_invocation.node_attrs.per_device_op_state);
  // Important: to make sure this doesn't get deallocated, we intentionally leak
  // the allocation here
  DeviceSpecificPerDeviceOpState *result_state_ptr =
      new DeviceSpecificPerDeviceOpState{result_state};
  spawn_device_state_init_return_task(ctx,
                                      task_args.origin_proc,
                                      *result_state_ptr,
                                      task_args.origin_result_ptr,
                                      Realm::Event::NO_EVENT);
}

std::optional<Realm::Event> spawn_device_state_init_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    ProfilingSettings const &profiling_settings,
    DeviceSpecificManagedPerDeviceFFHandle const &device_handle,
    FFIterationConfig const &iteration_config,
    OptimizerAttrs const &optimizer_attrs,
    DeviceSpecificPerDeviceOpState *result_ptr,
    Realm::Event precondition) {
  DeviceStateInitTaskArgs task_args{
      invocation,
      profiling_settings,
      device_handle,
      iteration_config,
      optimizer_attrs,
      ctx.get_current_processor(),
      result_ptr,
  };

  std::optional<task_id_t> task_id =
      and_then(and_then(invocation.node_attrs.op_attrs,
                        [](TrainingOperationAttrs const &op_attrs) {
                          return op_attrs.try_require_pcg_op();
                        }),
               get_init_task_id_for_op_attrs);
  if (task_id.has_value()) {
    std::string args = serialize_task_args(
        device_state_init_task_args_to_serializable(task_args));
    return ctx.spawn_task(target_proc,
                          assert_unwrap(task_id),
                          args.data(),
                          args.size(),
                          Realm::ProfilingRequestSet{},
                          precondition);
  }
  return std::nullopt;
}

} // namespace FlexFlow
