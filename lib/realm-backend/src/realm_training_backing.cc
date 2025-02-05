#include "local-execution/loss_functions.h"
#include "local-execution/optimizer.h"
#include "local-execution/task_id_t.dtg.h"
#include "local-execution/task_invocation.h"
#include "local-execution/task_signature_impl.h"
#include "local-execution/tensor_lowering.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "realm-backend/realm_training_backing.h"
#include "realm-backend/task_result.h"
#include "realm-backend/task_wrapper.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/values.h"
#include "utils/exception.h"

namespace FlexFlow {

using namespace Realm;

RealmTrainingBacking::RealmTrainingBacking(
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config, Realm::Processor master_proc)
    : computation_graph(computation_graph),
      realm_args_backing(runtime_arg_config),
      task_registry(empty_task_registry()) {
  master_proc = master_proc;
  proc_events.insert({master_proc, Realm::Event::NO_EVENT});
  master_mem = Machine::MemoryQuery(Machine::get_machine())
                   .only_kind(Memory::SYSTEM_MEM)
                   .best_affinity_to(master_proc)
                   .first();
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine())
                                   .only_kind(Processor::TOC_PROC);
  for (Processor p : pq) {
    worker_procs.push_back(p);
    proc_events.insert({p, Realm::Event::NO_EVENT});
    allocators.push_back(RealmAllocator(p));
  }
  assert(worker_procs.size() > 0);
}

void RealmTrainingBacking::register_and_allocate_layer(
    layer_guid_t const &node) {
  ComputationGraphOpAttrs attrs =
      get_layer_attrs(this->computation_graph, node).attrs;
  this->realm_tensor_backing.allocate_layer_tensors(
      node, this->computation_graph, this->allocators[0]);
  register_tasks_for_layer(this->task_registry, node, attrs);
  // TODO: multi gpu launching
  std::vector<task_id_t> task_ids = get_task_ids(attrs);
  for (task_id_t task_id : task_ids) {
    TaskSignatureAndImpl task_signature_impl =
        this->task_registry.task_mapping.at(task_id);
    register_wrapper_tasks(worker_procs[0], task_id, task_signature_impl);
  }
}

void RealmTrainingBacking::allocate_layer_optimizer_tensors(
    layer_guid_t const &node, OptimizerAttrs const &optimizer_attrs) {
  ComputationGraphOpAttrs attrs =
      get_layer_attrs(this->computation_graph, node).attrs;
  if (attrs.has<WeightAttrs>()) {
    TaskSignature sig = get_update_signature(optimizer_attrs);
    tensor_guid_t weight_tensor =
        get_only(get_outgoing_tensors(this->computation_graph, node));

    std::vector<optimizer_tensor_t> optimizer_tensors;
    for (TensorTypeSlotSpec const &tensor_type_slot_spec :
         values(sig.tensor_guid_slots)) {
      optimizer_tensors.push_back(
          this->optimizer_tensor_source.new_optimizer_tensor());
    }
    this->layer_optimizer_tensor_ids.insert({node, optimizer_tensors});
    this->realm_tensor_backing.allocate_optimizer_tensors(
        weight_tensor, optimizer_tensors, this->allocators[0]);
  }
}

void RealmTrainingBacking::execute_init(layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(this->task_registry, operator_node,
                                       OpTaskType::INIT)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;
    TaskInvocation invocation =
        this->lower_to_task_invocation(init(attrs), operator_node);
    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation);
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        this->task_registry.task_mapping.at(task_id).impl_function;
    // TODO: multi gpu launching
    Promise<DeviceSpecificDeviceStates> promise(master_mem);
    Future<DeviceSpecificDeviceStates> future = promise.get_future();
    RealmTaskArgs<DeviceSpecificDeviceStates> args{
        task_id, impl_function, accessor, std::move(promise)};
    Event e = worker_procs[0].spawn(static_cast<Processor::TaskFuncID>(task_id),
                                    &args, sizeof(args),
                                    proc_events[worker_procs[0]]);
    proc_events[worker_procs[0]] = e;
    future.set_event(e);
    this->realm_args_backing.add_per_device_op_state(operator_node,
                                                     std::move(future.get()));
  }
}

Future<std::optional<float>>
RealmTrainingBacking::execute_forward(layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(this->task_registry, operator_node,
                                       OpTaskType::FWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;
    TaskInvocation invocation =
        this->lower_to_task_invocation(forward(attrs), operator_node);
    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation);
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        this->task_registry.task_mapping.at(task_id).impl_function;
    // TODO: multi gpu launching
    Promise<std::optional<float>> promise(master_mem);
    Future<std::optional<float>> future = promise.get_future();
    RealmTaskArgs<std::optional<float>> args{task_id, impl_function, accessor,
                                             std::move(promise)};
    Event e = worker_procs[0].spawn(static_cast<Processor::TaskFuncID>(task_id),
                                    &args, sizeof(args),
                                    proc_events[worker_procs[0]]);
    proc_events[worker_procs[0]] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<std::optional<float>>(std::nullopt);
  }
}

Future<std::optional<float>>
RealmTrainingBacking::execute_backward(layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(this->task_registry, operator_node,
                                       OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;
    TaskInvocation invocation =
        this->lower_to_task_invocation(backward(attrs), operator_node);
    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation);
    task_id_t task_id = invocation.task_id;
    TaskImplFunction impl_function =
        this->task_registry.task_mapping.at(task_id).impl_function;
    // TODO: multi gpu launching
    Promise<std::optional<float>> promise(master_mem);
    Future<std::optional<float>> future = promise.get_future();
    RealmTaskArgs<std::optional<float>> args{task_id, impl_function, accessor,
                                             std::move(promise)};
    Event e = worker_procs[0].spawn(static_cast<Processor::TaskFuncID>(task_id),
                                    &args, sizeof(args),
                                    proc_events[worker_procs[0]]);
    proc_events[worker_procs[0]] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<std::optional<float>>(std::nullopt);
  }
}

Future<void>
RealmTrainingBacking::execute_update(layer_guid_t const &node,
                                     OptimizerAttrs const &optimizer_attrs) {
  LayerAttrs layer_attrs = get_layer_attrs(this->computation_graph, node);
  if (layer_attrs.attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor =
        get_only(get_outgoing_tensors(this->computation_graph, node));
    std::vector<optimizer_tensor_t> optimizer_buffer_tensors =
        this->layer_optimizer_tensor_ids.at(node);
    // get invocation
    TaskInvocation invocation = get_update_invocation(
        optimizer_attrs, weight_tensor, optimizer_buffer_tensors);
    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));
    TaskArgumentAccessor accessor = this->get_task_arg_accessor(invocation);
    task_id_t task_id = invocation.task_id;
    register_wrapper_tasks_generic(worker_procs[0], task_id);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    // TODO: multi gpu launching
    Promise<void> promise;
    Future<void> future = promise.get_future();
    RealmTaskArgs<void> args{task_id, update_impl_fn, accessor,
                             std::move(promise)};
    Event e = worker_procs[0].spawn(static_cast<Processor::TaskFuncID>(task_id),
                                    &args, sizeof(args),
                                    proc_events[worker_procs[0]]);
    proc_events[worker_procs[0]] = e;
    future.set_event(e);
    return future;
  } else {
    return Future<void>();
  }
}

Future<void>
RealmTrainingBacking::compute_loss(LossAttrs const &loss_attrs,
                                   tensor_guid_t const &logit_tensor,
                                   loss_tensor_t const &label_tensor) {
  TaskInvocation loss_invocation =
      backward(loss_attrs, logit_tensor, label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor =
      this->get_task_arg_accessor(loss_invocation);
  task_id_t task_id = loss_invocation.task_id;
  register_wrapper_tasks_generic(worker_procs[0], task_id);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  // TODO: multi gpu launching
  Promise<void> promise;
  Future<void> future = promise.get_future();
  RealmTaskArgs<void> args{task_id, loss_impl_fn, loss_accessor,
                           std::move(promise)};
  Event e =
      worker_procs[0].spawn(static_cast<Processor::TaskFuncID>(task_id), &args,
                            sizeof(args), proc_events[worker_procs[0]]);
  proc_events[worker_procs[0]] = e;
  future.set_event(e);
  return future;
}

TaskArgumentAccessor RealmTrainingBacking::get_task_arg_accessor(
    TaskInvocation const &invocation) const {
  TensorSlotsBacking tensor_slots_backing =
      this->realm_tensor_backing.construct_tensor_slots_backing(
          invocation.binding);
  ArgSlotsBacking arg_slots_backing =
      this->realm_args_backing.construct_arg_slots_backing(invocation.binding);
  return TaskArgumentAccessor::create<RealmTaskArgumentAccessor>(
      this->allocators[0], tensor_slots_backing, arg_slots_backing);
}

TaskInvocation RealmTrainingBacking::lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    layer_guid_t const &layer_guid) const {
  TaskBinding binding;
  // tensors
  for (auto const &tensor_binding :
       op_task_invocation.binding.get_tensor_bindings()) {
    tensor_guid_t tensor_to_bind = [&]() -> tensor_guid_t {
      switch (tensor_binding.second.role) {
      case TensorRole::INPUT:
        return get_incoming_inputs(this->computation_graph, layer_guid)
            .at(tensor_binding.second.idx);
      case TensorRole::OUTPUT:
        return get_outgoing_tensors(this->computation_graph, layer_guid)
            .at(tensor_binding.second.idx);
      case TensorRole::WEIGHT:
        return get_incoming_weights(this->computation_graph, layer_guid)
            .at(tensor_binding.second.idx);
      default:
        throw mk_runtime_error(
            fmt::format("Invalid tensor role {}", tensor_binding.second.role));
      }
    }();

    if (tensor_binding.first.is_grad == IsGrad::NO) {
      binding.bind(tensor_binding.first.slot_id, tensor_to_bind);
    } else if (tensor_binding.first.is_grad == IsGrad::YES) {
      binding.bind_grad(tensor_binding.first.slot_id, tensor_to_bind);
    } else {
      throw mk_runtime_error(fmt::format("Invalid value for IsGrad {}",
                                         tensor_binding.first.is_grad));
    }
  }

  // args
  for (auto const &arg_binding :
       op_task_invocation.binding.get_arg_bindings()) {
    if (arg_binding.second.has<OpArgRefSpec>()) {
      ConcreteArgSpec concrete_arg =
          this->realm_args_backing.lower_to_concrete_arg_spec(
              arg_binding.second.get<OpArgRefSpec>(), this->computation_graph,
              layer_guid);
      binding.insert_arg_spec(arg_binding.first, TaskArgSpec{concrete_arg});
    } else if (arg_binding.second.has<RuntimeArgRefSpec>()) {
      binding.insert_arg_spec(
          arg_binding.first,
          TaskArgSpec{arg_binding.second.get<RuntimeArgRefSpec>()});
    } else {
      binding.insert_arg_spec(
          arg_binding.first,
          TaskArgSpec{arg_binding.second.get<ConcreteArgSpec>()});
    }
  }

  return TaskInvocation{op_task_invocation.task_id, binding};
}

} // namespace FlexFlow
