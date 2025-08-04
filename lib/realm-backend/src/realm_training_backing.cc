#include "local-execution/local_args_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/loss_functions.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/optimizer.h"
#include "task-spec/task_invocation.h"
#include "task-spec/task_signature_impl.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "realm-backend/realm_training_backing.h"
#include "realm-backend/task_result.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using namespace Realm;

LocalTrainingBacking make_local_training_backing_for_computation_graph(
    RealmRuntimeState &runtime_state,
    std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated,
    TrainingComputationGraph const &training_computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs) {

  ASSERT(is_subseteq_of(                           
      keys(preallocated),
      keys(get_all_training_tensor_shapes(training_computation_graph))));

  LocalTaskRegistry local_task_registry =
      construct_local_task_registry_for_layers(get_layer_attrs_mapping(
          training_computation_graph.computation_graph));

  register_tasks_for_realm(local_task_registry, runtime_state);

  LocalTensorBacking local_tensor_backing = construct_local_tensor_backing(
      get_all_training_tensor_shapes(training_computation_graph),
      preallocated,
      runtime_state.allocators[0]);

  std::unordered_map<layer_guid_t, std::optional<DeviceSpecificDeviceStates>>
      per_device_op_states = generate_map(
          topological_ordering(training_computation_graph.computation_graph),
          [&](layer_guid_t const &layer_guid) {
            return create_per_device_op_state(
                local_task_registry,
                local_tensor_backing,
                runtime_arg_config,
                runtime_state,
                get_training_layer_plus_context(training_computation_graph,
                                                layer_guid));
          });

  LocalArgsBacking local_args_backing =
      make_local_args_backing_for_computation_graph(runtime_arg_config,
                                                    per_device_op_states);

  return LocalTrainingBacking{
      /*computation_graph=*/training_computation_graph,
      /*local_task_registry=*/local_task_registry,
      /*local_tensor_backing=*/local_tensor_backing,
      /*local_args_backing=*/local_args_backing,
  };
}

// register tasks for realm runtime
void register_tasks_for_realm(LocalTaskRegistry const &local_task_registry, RealmRuntimeState &runtime_state) {
    for (std::pair<task_id_t, TaskSignatureAndImpl> const &task : local_task_registry.task_mapping) {
        task_id_t task_id = task.first;
        TaskSignatureAndImpl task_signature_impl = task.second;
        // TODO: multi gpu
        register_wrapper_tasks(0, runtime_state.worker_procs[0], task_id, task_signature_impl);
    }
}

std::optional<DeviceSpecificDeviceStates>
    create_per_device_op_state(LocalTaskRegistry const &local_task_registry,
                               LocalTensorBacking const &tensor_backing,
                               RuntimeArgConfig const &runtime_arg_config,
                               RealmRuntimeState &runtime_state,
                               TrainingLayerPlusContext const &training_layer) {
  std::optional maybe_registered_task = try_get_registered_task(
      local_task_registry, training_layer.layer_guid, OpTaskType::INIT);

  ASSERT(maybe_registered_task.has_value());

  registered_task_t registered_task = maybe_registered_task.value();
  if (registered_task.is_noop_task()) {
    return std::nullopt;
  }

  TaskInvocation invocation = lower_to_task_invocation(
      /*op_task_invocation=*/get_init_op_task_invocation(
          training_layer.layer_attrs.op_attrs),
      /*training_layer=*/training_layer,
      /*device_specific_device_states=*/std::nullopt);

  TaskArgumentAccessor accessor = get_task_arg_accessor(
      tensor_backing, runtime_arg_config, invocation, runtime_state.allocators[0]);

  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      local_task_registry.task_mapping.at(task_id).impl_function;
  // TODO: multi gpu launching
  Promise<DeviceSpecificDeviceStates> promise = Promise<DeviceSpecificDeviceStates>();
  Future<DeviceSpecificDeviceStates> future = promise.get_future();
  RealmTaskArgs<DeviceSpecificDeviceStates>* task_arg = 
                        new RealmTaskArgs<DeviceSpecificDeviceStates>{
                            task_id, impl_function, accessor,
                            std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  Event e = runtime_state.worker_procs[0].spawn(
      get_realm_task_id(task_id), args, sizeof(uintptr_t),
      runtime_state.worker_events[0]);
  runtime_state.worker_events[0] = e;
  future.set_event(e);
  return future.get().value();
}

Future<std::optional<milliseconds_t>>
    execute_forward(LocalTaskRegistry const &local_task_registry,
                    LocalTensorBacking const &local_tensor_backing,
                    LocalArgsBacking const &local_args_backing,
                    TrainingLayerPlusContext const &training_layer,
                    RealmRuntimeState &runtime_state) {

  std::optional maybe_registered_task = try_get_registered_task(
      local_task_registry, training_layer.layer_guid, OpTaskType::BWD);

  ASSERT(maybe_registered_task.has_value());

  registered_task_t registered_task = maybe_registered_task.value();
  if (registered_task.is_noop_task()) {
    return Future<std::optional<milliseconds_t>>(std::nullopt);
  }

  std::optional<DeviceSpecificDeviceStates> device_state =
      get_per_device_op_state_if_exists(local_args_backing,
                                        training_layer.layer_guid);

  TaskInvocation invocation = lower_to_task_invocation(
      /*op_task_invocation=*/get_forward_op_task_invocation(
          training_layer.layer_attrs.op_attrs),
      /*training_layer=*/training_layer,
      /*device_specific_device_states=*/device_state);

  TaskArgumentAccessor accessor =
      get_task_arg_accessor(local_tensor_backing,
                            local_args_backing.runtime_arg_config,
                            invocation,
                            runtime_state.allocators[0]);

  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      local_task_registry.task_mapping.at(task_id).impl_function;
  // TODO: multi gpu launching
  Promise<std::optional<milliseconds_t>> promise(runtime_state.master_mem);
  Future<std::optional<milliseconds_t>> future = promise.get_future();
  RealmTaskArgs<std::optional<milliseconds_t>>* task_arg = 
                        new RealmTaskArgs<std::optional<milliseconds_t>>{
                            task_id, impl_function, accessor,
                            std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  Event e = runtime_state.worker_procs[0].spawn(
      get_realm_task_id(task_id), args, sizeof(uintptr_t),
      runtime_state.worker_events[0]);
  runtime_state.worker_events[0] = e;
  future.set_event(e);
  return future;
}

Future<std::optional<milliseconds_t>>
    execute_backward(LocalTaskRegistry const &local_task_registry,
                     LocalTensorBacking const &local_tensor_backing,
                     LocalArgsBacking const &local_args_backing,
                     TrainingLayerPlusContext const &training_layer,
                     RealmRuntimeState &runtime_state) {

  std::optional maybe_registered_task = try_get_registered_task(
      local_task_registry, training_layer.layer_guid, OpTaskType::BWD);

  ASSERT(maybe_registered_task.has_value());

  registered_task_t registered_task = maybe_registered_task.value();
  if (registered_task.is_noop_task()) {
    return Future<std::optional<milliseconds_t>>(std::nullopt);
  }

  std::optional<DeviceSpecificDeviceStates> device_state =
      get_per_device_op_state_if_exists(local_args_backing,
                                        training_layer.layer_guid);
  TaskInvocation invocation = lower_to_task_invocation(
      get_backward_op_task_invocation(training_layer.layer_attrs.op_attrs),
      training_layer,
      device_state);
  TaskArgumentAccessor accessor =
      get_task_arg_accessor(local_tensor_backing,
                            local_args_backing.runtime_arg_config,
                            invocation,
                            runtime_state.allocators[0]);

  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      local_task_registry.task_mapping.at(task_id).impl_function;
  // TODO: multi gpu launching
  Promise<std::optional<milliseconds_t>> promise(runtime_state.master_mem);
  Future<std::optional<milliseconds_t>> future = promise.get_future();
  RealmTaskArgs<std::optional<milliseconds_t>>* task_arg = 
                                new RealmTaskArgs<std::optional<milliseconds_t>>{
                                    task_id, impl_function, accessor,
                                    std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  Event e = runtime_state.worker_procs[0].spawn(
      get_realm_task_id(task_id), args, sizeof(uintptr_t),
      runtime_state.worker_events[0]);
  runtime_state.worker_events[0] = e;
  future.set_event(e);
  return future;
}

Future<void> execute_update(LocalTrainingBacking const &local_training_backing,
                    layer_guid_t const &layer_guid,
                    OptimizerAttrs const &optimizer_attrs,
                    RealmRuntimeState &runtime_state) {
  TrainingLayerPlusContext training_layer = get_training_layer_plus_context(
      local_training_backing.training_computation_graph, layer_guid);

  if (training_layer.layer_attrs.op_attrs.has<WeightAttrs>()) {
    TrainingTensorGroupWithAttrs weight_tensor_group =
        get_only(training_layer.output_tensor_groups);

    TaskInvocation invocation =
        get_update_invocation(optimizer_attrs,
                              weight_tensor_group.forward_tensor,
                              weight_tensor_group.gradient_tensor,
                              weight_tensor_group.optimizer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    TaskArgumentAccessor accessor = get_task_arg_accessor(
        local_training_backing.local_tensor_backing,
        local_training_backing.local_args_backing.runtime_arg_config,
        invocation,
        runtime_state.allocators[0]);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);

    task_id_t task_id = invocation.task_id;
    register_wrapper_tasks_generic(0, runtime_state.worker_procs[0],
                                   task_id);
    // TODO: multi gpu launching
    Promise<void> promise;
    Future<void> future = promise.get_future();
    RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{task_id, update_impl_fn, accessor,
                                        std::move(promise)};
    uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
    Event e = runtime_state.worker_procs[0].spawn(
        get_realm_task_id(task_id), args, sizeof(uintptr_t),
        runtime_state.worker_events[0]);
    runtime_state.worker_events[0] = e;
    future.set_event(e);
    return future;
  }
}

Future<void> compute_loss(LocalTrainingBacking const &local_training_backing,
                  LossAttrs const &loss_attrs,
                  RealmRuntimeState &runtime_state) {

  TrainingComputationGraph training_cg =
      local_training_backing.training_computation_graph;
  tensor_guid_t logit_tensor = training_cg.logit_tensor;
  loss_tensor_guid_t label_tensor = training_cg.label_tensor;

  TaskInvocation loss_invocation = backward(
      loss_attrs,
      get_forward_tensor_guid_for_tensor_guid(training_cg, logit_tensor),
      get_gradient_tensor_guid_for_tensor_guid(training_cg, logit_tensor),
      label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor = get_task_arg_accessor(
      local_training_backing.local_tensor_backing,
      local_training_backing.local_args_backing.runtime_arg_config,
      loss_invocation,
      runtime_state.allocators[0]);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();

  task_id_t task_id = loss_invocation.task_id;
  register_wrapper_tasks_generic(0, runtime_state.worker_procs[0],
                                task_id);
  // TODO: multi gpu launching
  Promise<void> promise;
  Future<void> future = promise.get_future();
  RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{task_id, loss_impl_fn, loss_accessor,
                                        std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  Event e = runtime_state.worker_procs[0].spawn(
      get_realm_task_id(task_id), args, sizeof(uintptr_t),
      runtime_state.worker_events[0]);
  runtime_state.worker_events[0] = e;
  future.set_event(e);
  return future;
}

} // namespace FlexFlow
