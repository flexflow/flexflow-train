#include "realm-execution/tasks/impl/device_state_init_task.h"
#include "kernels/device_handle_t.dtg.h"
#include "local-execution/device_state_initialization.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/tasks/impl/device_state_init_return_task.h"
#include "realm-execution/tasks/serializer/serializable_realm_processor.dtg.h"
#include "realm-execution/tasks/serializer/serializable_realm_processor.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "utils/optional.h"
#include <cstdint>
#include <optional>
#include <type_traits>

namespace FlexFlow {

// TODO: at some point we're going to have to actually serialize these, but for
// now just pass the pointer and assume we're running inside a single address
// space
struct DeviceStateInitTaskArgs {
  DeviceStateInitTaskArgs() = delete;
  DeviceStateInitTaskArgs(
      DynamicNodeInvocation const &invocation,
      ProfilingSettings const &profiling_settings,
      DeviceSpecificManagedPerDeviceFFHandle const &device_handle,
      FFIterationConfig const &iteration_config,
      OptimizerAttrs const &optimizer_attrs,
      Realm::Processor origin_proc,
      DeviceSpecificPerDeviceOpState *origin_result_ptr)
      : invocation(invocation), profiling_settings(profiling_settings),
        device_handle(device_handle), iteration_config(iteration_config),
        optimizer_attrs(optimizer_attrs), origin_proc(origin_proc),
        origin_result_ptr(origin_result_ptr) {}

  void serialize(nlohmann::json &j) const {
    nlohmann::json j_device_handle;
    device_handle.serialize(j_device_handle);
    j = {
        {"invocation", dynamic_node_invocation_to_serializable(invocation)},
        {"profiling_settings", profiling_settings},
        {"device_handle", j_device_handle},
        {"iteration_config", iteration_config},
        {"optimizer_attrs", optimizer_attrs},
        {"origin_proc", realm_processor_to_serializable(origin_proc)},
        {"origin_result_ptr", reinterpret_cast<uintptr_t>(origin_result_ptr)},
    };
  }

  static DeviceStateInitTaskArgs deserialize(nlohmann::json const &j) {
    return DeviceStateInitTaskArgs{
        /*invocation=*/dynamic_node_invocation_from_serializable(
            j.at("invocation").get<SerializableDynamicNodeInvocation>()),
        /*profiling_settings=*/
        j.at("profiling_settings").get<ProfilingSettings>(),
        /*device_handle=*/
        DeviceSpecificManagedPerDeviceFFHandle::deserialize(
            j.at("device_handle")),
        /*iteration_config=*/j.at("iteration_config").get<FFIterationConfig>(),
        /*optimizer_attrs=*/j.at("optimizer_attrs").get<OptimizerAttrs>(),
        /*origin_proc=*/
        realm_processor_from_serializable(
            j.at("origin_proc").get<SerializableRealmProcessor>()),
        /*origin_result_ptr=*/
        reinterpret_cast<DeviceSpecificPerDeviceOpState *>(
            j.at("origin_result_ptr").get<uintptr_t>()),
    };
  }

public:
  DynamicNodeInvocation invocation;
  ProfilingSettings profiling_settings;
  DeviceSpecificManagedPerDeviceFFHandle device_handle;
  FFIterationConfig iteration_config;
  OptimizerAttrs optimizer_attrs;
  Realm::Processor origin_proc;
  DeviceSpecificPerDeviceOpState *origin_result_ptr;
};

void device_state_init_task_body(void const *args,
                                 size_t arglen,
                                 void const *userdata,
                                 size_t userlen,
                                 Realm::Processor proc) {
  DeviceStateInitTaskArgs task_args =
      deserialize_task_args<DeviceStateInitTaskArgs>(args, arglen);

  // FIXME: serialize instead of passing pointers around
  ASSERT(task_args.origin_proc.address_space() == proc.address_space());

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
    std::string args = serialize_task_args(task_args);
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
