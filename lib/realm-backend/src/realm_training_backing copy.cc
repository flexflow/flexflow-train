// #include "kernels/allocation.h"
// #include "local-execution/loss_functions.h"
// #include "local-execution/optimizer.h"
// #include "pcg/computation_graph.dtg.h"
// #include "pcg/computation_graph.h"
// #include "pcg/optimizer_attrs.h"
// #include "realm-backend/realm_tensor_backing.h"
// #include "task-spec/op_task_to_task_invocation.h"
// #include "task-spec/runtime_arg_config.h"
// #include "task-spec/task_invocation.h"
// #include "task-spec/task_signature_impl.h"
// #include "utils/containers/contains.h"
// #include "utils/containers/contains_key.h"
// #include "utils/containers/get_only.h"
// #include "utils/containers/values.h"
// #include "utils/exception.h"

// #include "realm-backend/realm_training_backing.h"
// #include "realm-backend/task_result.h"
// #include "realm-backend/task_wrapper.h"

// namespace FlexFlow {

// using namespace Realm;

// RealmTrainingBacking::RealmTrainingBacking(
//     Processor master_proc, std::vector<Processor> const &worker_procs,
//     std::vector<Allocator> const &allocators,
//     AllocatedTensors const &allocated_tensors,
//     GradientTensorSource &gradient_tensor_source,
//     ComputationGraph const &computation_graph,
//     RuntimeArgConfig const &runtime_arg_config)
//     : master_proc(master_proc), master_event(Realm::Event::NO_EVENT),
//       master_mem(Machine::MemoryQuery(Machine::get_machine())
//                      .only_kind(Memory::SYSTEM_MEM)
//                      .best_affinity_to(master_proc)
//                      .first()),
//     worker_procs(worker_procs),
//     worker_events(std::vector<Realm::Event>(worker_procs.size(),
//                                            Realm::Event::NO_EVENT)),
//       allocators(allocators), computation_graph(computation_graph),
//       task_registry(construct_task_registry_and_register_tasks_for_realm(
//           computation_graph, worker_procs)),
//       realm_tensor_backing(construct_realm_tensor_backing( // TODO: multi gpu
//         allocated_tensors,
//         generate_unallocated_tensors(
//             allocated_tensors, get_all_tensor_attrs(computation_graph),
//             gradient_tensor_source),
//         this->allocators[0])),
//       realm_args_backing(initialize_args_backing(this, computation_graph, runtime_arg_config)) {}

// TaskRegistry construct_task_registry_and_register_tasks_for_realm(
//     ComputationGraph const &cg, std::vector<Realm::Processor> const &worker_procs) {
//   TaskRegistry task_registry = construct_task_registry(
//     get_layer_attrs_mapping(cg));

//   // register tasks for realm
//   std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping =
//       get_layer_attrs_mapping(cg);
//   for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs :
//       layer_attrs_mapping) {
//     ComputationGraphOpAttrs attrs = layer_attrs.second.op_attrs;
//     std::vector<task_id_t> task_ids = get_task_ids(attrs);
//     for (task_id_t task_id : task_ids) {
//         TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
//         // TODO: multi gpu
//         register_wrapper_tasks(0, worker_procs[0], task_id, task_signature_impl);
//     }
//   }

//   return task_registry;
// }

// RealmArgsBacking
// initialize_args_backing(RealmTrainingBacking *backing,
//                         ComputationGraph const &cg,
//                         RuntimeArgConfig const &runtime_arg_config) {
//   std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
//       per_device_op_states;
//   TaskRegistry const &task_registry = backing->task_registry;
//   RealmTensorBacking const &realm_tensor_backing =
//       backing->realm_tensor_backing;
//   Processor master_proc = backing->master_proc;
//   Memory master_mem = backing->master_mem;
//   std::vector<Processor> &worker_procs = backing->worker_procs;
//   std::vector<Event> &worker_events = backing->worker_events;
//   // TODO: multi gpu
//   Allocator &allocator = backing->allocators[0];

//   for (layer_guid_t const &node : topological_ordering(cg)) {
//     if (registry_contains_task_for_layer(task_registry, node,
//                                          OpTaskType::INIT)) {
//       ComputationGraphOpAttrs attrs = get_layer_attrs(cg, node).op_attrs;

//       TaskInvocation invocation = lower_to_task_invocation(
//           init(attrs), node, get_incoming_inputs(cg, node),
//           get_incoming_input_shapes(cg, node), get_outgoing_tensors(cg, node),
//           get_incoming_weights(cg, node),
//           realm_tensor_backing.tensor_gradient_mapping, std::nullopt);
//       TaskArgumentAccessor accessor = get_task_arg_accessor(
//           realm_tensor_backing,
//           make_args_backing_with_empty_device_states(runtime_arg_config),
//           invocation,
//           allocator);
//       task_id_t task_id = invocation.task_id;
//       TaskImplFunction impl_function =
//           task_registry.task_mapping.at(task_id).impl_function;
//       // TODO: multi gpu launching
//       Promise<DeviceSpecificDeviceStates> promise = Promise<DeviceSpecificDeviceStates>();
//       Future<DeviceSpecificDeviceStates> future = promise.get_future();
//       RealmTaskArgs<DeviceSpecificDeviceStates>* task_arg = new RealmTaskArgs<DeviceSpecificDeviceStates>{
//           task_id, impl_function, accessor, std::move(promise)};
//       uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
//       Event e =
//           worker_procs[0].spawn(get_realm_task_id(task_id),
//                                 args, sizeof(uintptr_t), worker_events[0]);
//       worker_events[0] = e;
//       future.set_event(e);
//       per_device_op_states.insert({node, future.get().value()});
//     }
//   }

//   return RealmArgsBacking{runtime_arg_config, per_device_op_states};
// }

// } // namespace FlexFlow
