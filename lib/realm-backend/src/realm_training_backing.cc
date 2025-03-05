#include "kernels/allocation.h"
#include "local-execution/loss_functions.h"
#include "local-execution/optimizer.h"
#include "local-execution/task_signature_impl.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "realm-backend/realm_tensor_backing.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/runtime_arg_config.h"
#include "task-spec/task_invocation.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/values.h"
#include "utils/exception.h"

#include "realm-backend/realm_training_backing.h"
#include "realm-backend/task_result.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using namespace Realm;

RealmTrainingBacking::RealmTrainingBacking(
    Processor master_proc, std::vector<Processor> const &worker_procs,
    std::vector<Allocator> const &allocators,
    AllocatedTensors const &allocated_tensors,
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config)
    : master_proc(master_proc), worker_procs(worker_procs),
      allocators(allocators), computation_graph(computation_graph),
      task_registry(construct_task_registry(
          get_layer_attrs_mapping(this->computation_graph))),
      realm_tensor_backing(construct_realm_tensor_backing( // TODO: multi gpu
        allocated_tensors,
        generate_unallocated_tensors(
            allocated_tensors, get_all_tensor_attrs(this->computation_graph),
            this->gradient_tensor_source),
        this->allocators[0])),
      realm_args_backing(initialize_args_backing(this, runtime_arg_config)) {
  master_event = Realm::Event::NO_EVENT;
  master_mem = Machine::MemoryQuery(Machine::get_machine())
                   .only_kind(Memory::SYSTEM_MEM)
                   .best_affinity_to(master_proc)
                   .first();
  for (Processor p : worker_procs) {
    worker_events.push_back(Realm::Event::NO_EVENT);
  }
  //   Machine::ProcessorQuery pq =
  //   Machine::ProcessorQuery(Machine::get_machine())
  //                                    .only_kind(Processor::TOC_PROC);
  // allocators.push_back(create_realm_memory_allocator(p));

  // register tasks for realm
  std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping =
      get_layer_attrs_mapping(this->computation_graph);
  for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs :
      layer_attrs_mapping) {
    ComputationGraphOpAttrs attrs = layer_attrs.second.attrs;
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
        TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
        // TODO: multi gpu
        register_wrapper_tasks(worker_procs[0], task_id, task_signature_impl);
    }
  }
}

RealmTrainingBacking::RealmTrainingBacking(
    Processor master_proc, std::vector<Processor> const &worker_procs,
    std::vector<Allocator> const &allocators,
    AllocatedTensors const &allocated_tensors,
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs)
    : master_proc(master_proc), worker_procs(worker_procs),
      allocators(allocators), computation_graph(computation_graph),
      task_registry(construct_task_registry(
          get_layer_attrs_mapping(this->computation_graph))),
    realm_tensor_backing(construct_realm_tensor_backing( // TODO: multi gpu
        allocated_tensors,
        generate_unallocated_tensors_with_optimizer(
            allocated_tensors, get_all_tensor_attrs(this->computation_graph),
            this->gradient_tensor_source, this->optimizer_tensor_source,
            optimizer_attrs),
        this->allocators[0])),
      realm_args_backing(initialize_args_backing(this, runtime_arg_config)) {
  master_event = Realm::Event::NO_EVENT;
  master_mem = Machine::MemoryQuery(Machine::get_machine())
                   .only_kind(Memory::SYSTEM_MEM)
                   .best_affinity_to(master_proc)
                   .first();
  for (Processor p : worker_procs) {
    worker_events.push_back(Realm::Event::NO_EVENT);
  }

  // register tasks for realm
  std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping =
      get_layer_attrs_mapping(this->computation_graph);
  for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs :
      layer_attrs_mapping) {
    ComputationGraphOpAttrs attrs = layer_attrs.second.attrs;
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
        TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
        // TODO: multi gpu
        register_wrapper_tasks(worker_procs[0], task_id, task_signature_impl);
    }
  }
}

RealmArgsBacking
initialize_args_backing(RealmTrainingBacking *backing,
                        RuntimeArgConfig const &runtime_arg_config) {
  // initialize_args_backing(TaskRegistry const &task_registry,
  //                         ComputationGraph const &cg,
  //                         RuntimeArgConfig const &runtime_arg_config,
  //                         RealmTensorBacking const &realm_tensor_backing) {
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
      per_device_op_states;
  TaskRegistry const &task_registry = backing->task_registry;
  ComputationGraph const &cg = backing->computation_graph;
  RealmTensorBacking const &realm_tensor_backing =
      backing->realm_tensor_backing;
  Processor master_proc = backing->master_proc;
  Memory master_mem = backing->master_mem;
  std::vector<Processor> &worker_procs = backing->worker_procs;
  std::vector<Event> &worker_events = backing->worker_events;
  // TODO: multi gpu
  Allocator &allocator = backing->allocators[0];

  for (layer_guid_t const &node : topological_ordering(cg)) {
    if (registry_contains_task_for_layer(task_registry, node,
                                         OpTaskType::INIT)) {
      ComputationGraphOpAttrs attrs = get_layer_attrs(cg, node).attrs;

      TaskInvocation invocation = lower_to_task_invocation(
          init(attrs), node, get_incoming_inputs(cg, node),
          get_incoming_input_shapes(cg, node), get_outgoing_tensors(cg, node),
          get_incoming_weights(cg, node),
          realm_tensor_backing.tensor_gradient_mapping, std::nullopt);
      TaskArgumentAccessor accessor = get_task_arg_accessor(
          realm_tensor_backing,
          make_args_backing_with_empty_device_states(runtime_arg_config),
          invocation,
          allocator);
      task_id_t task_id = invocation.task_id;
      TaskImplFunction impl_function =
          task_registry.task_mapping.at(task_id).impl_function;
      // TODO: multi gpu launching
      Promise<DeviceSpecificDeviceStates> promise(master_mem);
      Future<DeviceSpecificDeviceStates> future = promise.get_future();
      RealmTaskArgs<DeviceSpecificDeviceStates> args{
          task_id, impl_function, accessor, std::move(promise)};
      Event e =
          worker_procs[0].spawn(static_cast<Processor::TaskFuncID>(task_id),
                                &args, sizeof(args), worker_events[0]);
      worker_events[0] = e;
      future.set_event(e);
      per_device_op_states.insert({node, std::move(future.get())});
    }
  }

  return RealmArgsBacking{runtime_arg_config, per_device_op_states};
}

Future<float>
execute_forward(RealmTrainingBacking &realm_training_backing,
                layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(realm_training_backing.task_registry,
                                       operator_node, OpTaskType::FWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(realm_training_backing.computation_graph, operator_node)
            .attrs;
    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            realm_training_backing.realm_args_backing, operator_node);
    TaskInvocation invocation = lower_to_task_invocation(
        forward(attrs), operator_node,
        get_incoming_inputs(realm_training_backing.computation_graph,
                            operator_node),
        get_incoming_input_shapes(realm_training_backing.computation_graph,
                                  operator_node),
        get_outgoing_tensors(realm_training_backing.computation_graph,
                             operator_node),
        get_incoming_weights(realm_training_backing.computation_graph,
                             operator_node),
        realm_training_backing.realm_tensor_backing.tensor_gradient_mapping,
        device_state);
    TaskArgumentAccessor accessor = get_task_arg_accessor(
        realm_training_backing.realm_tensor_backing,
        realm_training_backing.realm_args_backing, invocation,
        realm_training_backing.allocators[0]);
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        realm_training_backing.task_registry.task_mapping.at(task_id)
            .impl_function;
    // TODO: multi gpu launching
    Promise<float> promise(realm_training_backing.master_mem);
    Future<float> future = promise.get_future();
    RealmTaskArgs<float> args{task_id, impl_function, accessor,
                                std::move(promise)};
    Event e = realm_training_backing.worker_procs[0].spawn(
        static_cast<Processor::TaskFuncID>(task_id), &args, sizeof(args),
        realm_training_backing.worker_events[0]);
    realm_training_backing.worker_events[0] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<float>(0.0f);
  }
}

Future<float>
execute_backward(RealmTrainingBacking &realm_training_backing,
                 layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(realm_training_backing.task_registry,
                                       operator_node, OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(realm_training_backing.computation_graph, operator_node)
            .attrs;
    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            realm_training_backing.realm_args_backing, operator_node);
    TaskInvocation invocation = lower_to_task_invocation(
        forward(attrs), operator_node,
        get_incoming_inputs(realm_training_backing.computation_graph,
                            operator_node),
        get_incoming_input_shapes(realm_training_backing.computation_graph,
                                  operator_node),
        get_outgoing_tensors(realm_training_backing.computation_graph,
                             operator_node),
        get_incoming_weights(realm_training_backing.computation_graph,
                             operator_node),
        realm_training_backing.realm_tensor_backing.tensor_gradient_mapping,
        device_state);
    TaskArgumentAccessor accessor = get_task_arg_accessor(
        realm_training_backing.realm_tensor_backing,
        realm_training_backing.realm_args_backing, invocation,
        realm_training_backing.allocators[0]);
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        realm_training_backing.task_registry.task_mapping.at(task_id)
            .impl_function;
    // TODO: multi gpu launching
    Promise<float> promise(realm_training_backing.master_mem);
    Future<float> future = promise.get_future();
    RealmTaskArgs<float> args{task_id, impl_function, accessor,
                                std::move(promise)};
    Event e = realm_training_backing.worker_procs[0].spawn(
        static_cast<Processor::TaskFuncID>(task_id), &args, sizeof(args),
        realm_training_backing.worker_events[0]);
    realm_training_backing.worker_events[0] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<float>(0.0f);
  }
}

Future<void> execute_update(RealmTrainingBacking &realm_training_backing,
                            layer_guid_t const &node,
                            OptimizerAttrs const &optimizer_attrs) {
  LayerAttrs layer_attrs =
      get_layer_attrs(realm_training_backing.computation_graph, node);
  if (layer_attrs.op_attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor = get_only(
        get_outgoing_tensors(realm_training_backing.computation_graph, node));

    gradient_tensor_t weight_grad_tensor =
        realm_training_backing.realm_tensor_backing.tensor_gradient_mapping.at(
            weight_tensor);
    std::vector<optimizer_tensor_t> optimizer_buffer_tensors =
        realm_training_backing.realm_tensor_backing.tensor_optimizer_mapping.at(
            weight_tensor);

    // get invocation
    TaskInvocation invocation =
        get_update_invocation(optimizer_attrs, weight_tensor,
                              weight_grad_tensor, optimizer_buffer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    // execute update
    TaskArgumentAccessor accessor = get_task_arg_accessor(
        realm_training_backing.realm_tensor_backing,
        realm_training_backing.realm_args_backing, invocation,
        realm_training_backing.allocators[0]);
    task_id_t task_id = invocation.task_id;
    register_wrapper_tasks_generic(realm_training_backing.worker_procs[0],
                                   task_id);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    // TODO: multi gpu launching
    Promise<void> promise;
    Future<void> future = promise.get_future();
    RealmTaskArgs<void> args{task_id, update_impl_fn, accessor,
                             std::move(promise)};
    Event e = realm_training_backing.worker_procs[0].spawn(
        static_cast<Processor::TaskFuncID>(task_id), &args, sizeof(args),
        realm_training_backing.worker_events[0]);
    realm_training_backing.worker_events[0] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<void>();
  }
}

Future<void> compute_loss(RealmTrainingBacking &realm_training_backing,
                          LossAttrs const &loss_attrs,
                          tensor_guid_t const &logit_tensor,
                          loss_tensor_t const &label_tensor) {
  TaskInvocation loss_invocation = backward(
      loss_attrs, logit_tensor,
      realm_training_backing.realm_tensor_backing.tensor_gradient_mapping.at(
          logit_tensor),
      label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor = get_task_arg_accessor(
      realm_training_backing.realm_tensor_backing,
      realm_training_backing.realm_args_backing, loss_invocation,
        realm_training_backing.allocators[0]);
  task_id_t task_id = loss_invocation.task_id;
  register_wrapper_tasks_generic(realm_training_backing.worker_procs[0],
                                 task_id);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  // TODO: multi gpu launching
  Promise<void> promise;
  Future<void> future = promise.get_future();
  RealmTaskArgs<void> args{task_id, loss_impl_fn, loss_accessor,
                           std::move(promise)};
  Event e = realm_training_backing.worker_procs[0].spawn(
      static_cast<Processor::TaskFuncID>(task_id), &args, sizeof(args),
      realm_training_backing.worker_events[0]);
  realm_training_backing.worker_events[0] = e;
  future.set_event(e);
  return future;
}

TaskArgumentAccessor
get_task_arg_accessor(RealmTensorBacking const &realm_tensor_backing,
                      RealmArgsBacking const &realm_args_backing,
                      TaskInvocation const &invocation,
                      Allocator &allocator) {
  TensorSlotsBacking tensor_slots_backing =
      construct_tensor_slots_backing(realm_tensor_backing, invocation.binding);
  ArgSlotsBacking arg_slots_backing = construct_arg_slots_backing(
      invocation.binding, realm_args_backing.runtime_arg_config);
  // TODO: multi gpu
  return TaskArgumentAccessor::create<RealmTaskArgumentAccessor>(
      allocator, tensor_slots_backing, arg_slots_backing);
}

} // namespace FlexFlow
