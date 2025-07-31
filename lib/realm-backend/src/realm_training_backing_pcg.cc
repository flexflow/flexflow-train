#include "realm-backend/realm_training_backing_pcg.h"
#include "realm-backend/realm_training_backing.h"
#include "compiler/machine_mapping/unstructured_device_mapping.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/operator_task_space.h"
#include "pcg/machine_view.h"
#include "compiler/task_graph_simulator/pcg_task_graph.h"
#include "utils/containers/get_only.h"
#include "pcg/gpu_id_t.dtg.h"
#include "utils/integer_types.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/parallel_tensor_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/transform.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "op-attrs/operator_type.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/overload.h"
#include <unordered_set>
#include <cstring>  // For memcpy

namespace FlexFlow {

using namespace Realm;

// Parallelization strategy types
enum class ParallelizationType {
  DATA_PARALLEL,       
  MODEL_PARALLEL,      
  PIPELINE_PARALLEL,  
  HYBRID_PARALLEL      
};

// Parallelization strategy configuration
struct ParallelizationStrategy {
  ParallelizationType type;
  size_t partition_size;  // For model parallelism
  size_t stage_id;        // For pipeline parallelism
  
  ParallelizationStrategy(ParallelizationType t = ParallelizationType::DATA_PARALLEL,
                         size_t ps = 1, size_t sid = 0)
    : type(t), partition_size(ps), stage_id(sid) {}
};

// Parallel execution context for device-specific task execution
struct ParallelExecutionContext {
  RealmTrainingBackingPCG &backing;
  parallel_layer_guid_t layer;
  device_id_t device;
  PCGOperatorAttrs op_attrs;
  TensorShape device_input_shape; 
  
  ParallelExecutionContext(RealmTrainingBackingPCG &b, 
                          parallel_layer_guid_t l,
                          device_id_t d,
                          PCGOperatorAttrs attrs)
    : backing(b), layer(l), device(d), op_attrs(attrs) {}
};

// Helper: Create task invocation for specific device
TaskInvocation create_task_invocation_for_device(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    device_id_t device,
    PCGOperatorAttrs const &attrs,
    std::optional<TensorShape> device_input_shape = std::nullopt) {
  
  OpTaskInvocation op_invocation = forward(attrs);
  
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
  
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  std::vector<TensorShape> input_shapes;
  if (device_input_shape.has_value()) {
    // Use device-specific shape for data parallel
    input_shapes.push_back(device_input_shape.value());
  } else {
    // Use original shapes from PCG
    for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
      ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
      input_shapes.push_back(get_piece_shape(parallel_shape));
    }
  }
  
  // Get device states if available
  std::optional<DeviceSpecificDeviceStates> device_state = 
      get_per_device_op_state_if_exists(backing.realm_args_backing, regular_layer);
  
  // Convert OpTaskInvocation to TaskInvocation
  return lower_to_task_invocation(
      op_invocation,
      regular_layer,
      inputs,
      input_shapes,
      outputs,
      weights,
      backing.get_device_tensor_backing(device).tensor_gradient_mapping,  // Use device-specific backing
      device_state);
}

// Multi-GPU tensor backing construction - create device-specific backings
std::unordered_map<device_id_t, RealmTensorBacking> construct_device_specific_tensor_backings(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    std::vector<Allocator> const &allocators,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec,
    ParallelComputationGraph const &pcg) {
  
  if (allocators.empty()) {
    throw std::runtime_error("No allocators provided for multi-GPU tensor backing");
  }
  
  std::unordered_map<device_id_t, RealmTensorBacking> device_tensor_backings;
  
  try {
    // Get device mapping from PCG
    UnstructuredDeviceMapping device_mapping = 
        get_unstructured_device_mapping(machine_mapping, machine_spec, pcg);
    
    std::unordered_map<tensor_guid_t, device_id_t> tensor_device_mapping = 
        create_tensor_device_mapping(pcg, device_mapping, allocators.size());
    
    // Create device-specific tensor backings with PHYSICAL replication
    for (size_t i = 0; i < allocators.size(); i++) {
      device_id_t device = device_id_t(gpu_id_t(nonnegative_int(i)));
      
      AllocatedTensors device_allocated = replicate_tensors_for_device(
          allocated_tensors, device, const_cast<Allocator&>(allocators[i]));
      UnallocatedTensors device_unallocated = replicate_unallocated_tensors_for_device(
          unallocated_tensors, device, const_cast<Allocator&>(allocators[i]));
      
      RealmTensorBacking device_backing = construct_realm_tensor_backing(
          device_allocated, device_unallocated, 
          const_cast<Allocator&>(allocators[i]));
      
      device_tensor_backings[device] = device_backing;
    }
    
  } catch (const std::exception& e) {
    // Fallback: create device-specific backings with physical replication
    for (size_t i = 0; i < allocators.size(); i++) {
      device_id_t device = device_id_t(gpu_id_t(nonnegative_int(i)));
      
      Allocator &primary_allocator = const_cast<Allocator&>(allocators[0]);
      AllocatedTensors device_allocated = replicate_tensors_for_device(
          allocated_tensors, device, primary_allocator);
      UnallocatedTensors device_unallocated = replicate_unallocated_tensors_for_device(
          unallocated_tensors, device, primary_allocator);
      
      RealmTensorBacking device_backing = construct_realm_tensor_backing(
          device_allocated, device_unallocated, primary_allocator);
      
      device_tensor_backings[device] = device_backing;
    }
  }
  
  return device_tensor_backings;
}

// Helper: Create tensor-to-device mapping based on PCG 
std::unordered_map<tensor_guid_t, device_id_t> create_tensor_device_mapping(
    ParallelComputationGraph const &pcg,
    UnstructuredDeviceMapping const &device_mapping,
    size_t num_devices) {
  
  std::unordered_map<tensor_guid_t, device_id_t> mapping;
  
  // Get all tensors from PCG
  std::unordered_set<parallel_tensor_guid_t> parallel_tensors = get_parallel_tensors(pcg);
  
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_tensors) {
    try {
      tensor_guid_t tensor = convert_parallel_to_regular_tensor(parallel_tensor);
      
      device_id_t device = get_tensor_device_placement(device_mapping, parallel_tensor);

      if (device.gpu_id.gpu_index.raw_value < num_devices) {
        mapping[tensor] = device;
      } else {
        size_t device_index = std::hash<tensor_guid_t>{}(tensor) % num_devices;
        mapping[tensor] = device_id_t(gpu_id_t(nonnegative_int(device_index)));
      }
      
    } catch (const std::exception& e) {
      continue;
    }
  }
  
  return mapping;
}

// Helper: Filter allocated tensors for specific device
AllocatedTensors filter_tensors_for_device(
    AllocatedTensors const &all_tensors,
    std::unordered_map<tensor_guid_t, device_id_t> const &tensor_device_mapping,
    device_id_t target_device) {
  
  AllocatedTensors device_tensors;
  
  for (auto const &tensor_pair : all_tensors) {
    tensor_guid_t tensor_guid = tensor_pair.first;
    
    // Check if this tensor is assigned to the target device
    auto it = tensor_device_mapping.find(tensor_guid);
    if (it != tensor_device_mapping.end() && it->second == target_device) {
      device_tensors[tensor_guid] = tensor_pair.second;
    }
  }
  
  return device_tensors;
}

// Helper: Filter unallocated tensors for specific device
UnallocatedTensors filter_unallocated_tensors_for_device(
    UnallocatedTensors const &all_tensors,
    std::unordered_map<tensor_guid_t, device_id_t> const &tensor_device_mapping,
    device_id_t target_device) {
  
  UnallocatedTensors device_tensors;
  
  for (auto const &tensor_pair : all_tensors) {
    tensor_guid_t tensor_guid = tensor_pair.first;
    
    // Check if this tensor is assigned to the target device
    auto it = tensor_device_mapping.find(tensor_guid);
    if (it != tensor_device_mapping.end() && it->second == target_device) {
      device_tensors[tensor_guid] = tensor_pair.second;
    }
  }
  
  return device_tensors;
}

RealmTrainingBackingPCG::RealmTrainingBackingPCG(
    Processor master_proc, 
    std::vector<Processor> const &worker_procs,
    std::vector<Allocator> const &allocators,
    AllocatedTensors const &allocated_tensors,
    GradientTensorSource &gradient_tensor_source,
    ParallelComputationGraph const &pcg, // additional pcg parameter
    MachineMapping const &machine_mapping, 
    MachineSpecification const &machine_spec,
    RuntimeArgConfig const &runtime_arg_config)
  : master_proc(master_proc), 
    master_event(Event::NO_EVENT),
    master_mem(Machine::MemoryQuery(Machine::get_machine())
                  .only_kind(Memory::SYSTEM_MEM)
                  .best_affinity_to(master_proc)
                  .first()),
    worker_procs(worker_procs),
    worker_events(std::vector<Event>(worker_procs.size(), Event::NO_EVENT)),
    allocators(allocators),
    pcg(pcg),
    machine_mapping(machine_mapping),
    machine_spec(machine_spec),
    task_registry(construct_task_registry_and_register_tasks_for_realm_pcg(pcg, worker_procs)),
    device_tensor_backings(construct_device_specific_tensor_backings(
        allocated_tensors,
        generate_unallocated_tensors(
            allocated_tensors, get_all_tensor_attrs_from_pcg(pcg),
            gradient_tensor_source),
        allocators,  // Pass all allocators for multi-GPU distribution
        machine_mapping, machine_spec, pcg)),
    realm_args_backing(initialize_args_backing_pcg(this, pcg, runtime_arg_config)) {
  
  initialize_device_mappings();
}

RealmTrainingBackingPCG::RealmTrainingBackingPCG(
    Processor master_proc,
    std::vector<Processor> const &worker_procs,
    std::vector<Allocator> const &allocators,
    AllocatedTensors const &allocated_tensors,
    GradientTensorSource &gradient_tensor_source,
    OptimizerTensorSource &optimizer_tensor_source,
    ParallelComputationGraph const &pcg,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs)
  : master_proc(master_proc),
    master_event(Event::NO_EVENT),
    master_mem(Machine::MemoryQuery(Machine::get_machine())
                  .only_kind(Memory::SYSTEM_MEM)
                  .best_affinity_to(master_proc)
                  .first()),
    worker_procs(worker_procs),
    worker_events(std::vector<Event>(worker_procs.size(), Event::NO_EVENT)),
    allocators(allocators),
    pcg(pcg),
    machine_mapping(machine_mapping),
    machine_spec(machine_spec),
    task_registry(construct_task_registry_and_register_tasks_for_realm_pcg(pcg, worker_procs)),
    device_tensor_backings(construct_device_specific_tensor_backings(
        allocated_tensors,
        generate_unallocated_tensors_with_optimizer(
            allocated_tensors, get_all_tensor_attrs_from_pcg(pcg),
            gradient_tensor_source, optimizer_tensor_source,
            optimizer_attrs),
        allocators,  // Pass all allocators for multi-GPU distribution
        machine_mapping, machine_spec, pcg)),
    realm_args_backing(initialize_args_backing_pcg(this, pcg, runtime_arg_config)) {
  
  initialize_device_mappings();
}

void RealmTrainingBackingPCG::initialize_device_mappings() {
  UnstructuredDeviceMapping device_mapping = 
      get_unstructured_device_mapping(machine_mapping, machine_spec, pcg);
  
  // Build device-to-processor mapping
  // Multi-GPU: Create device mappings for all available processors
  size_t num_devices = std::min(worker_procs.size(), allocators.size());
  
  for (size_t i = 0; i < num_devices; i++) {
    device_id_t device = device_id_t(gpu_id_t(nonnegative_int(i)));
    
    // Map each device to a corresponding processor
    device_to_processor[device] = worker_procs[i % worker_procs.size()];
    
  }
}

TaskRegistry construct_task_registry_and_register_tasks_for_realm_pcg(
    ParallelComputationGraph const &pcg,
    std::vector<Processor> const &worker_procs) {
  
  std::unordered_map<layer_guid_t, LayerAttrs> layer_attrs_mapping = 
      get_layer_attrs_mapping_from_pcg(pcg);
  
  TaskRegistry task_registry = construct_task_registry(layer_attrs_mapping);

  // Register tasks for realm - similar to classic version
  for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs : layer_attrs_mapping) {
    ComputationGraphOpAttrs attrs = layer_attrs.second.op_attrs;
    std::vector<task_id_t> task_ids = get_task_ids(attrs);
    for (task_id_t task_id : task_ids) {
        TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
        // Register for all available processors (multi-GPU support)
        for (size_t i = 0; i < worker_procs.size(); i++) {
            register_wrapper_tasks(i, worker_procs[i], task_id, task_signature_impl);
        }
    }
  }
  
  return task_registry;
}

RealmArgsBacking initialize_args_backing_pcg(
    RealmTrainingBackingPCG *backing,
    ParallelComputationGraph const &pcg,
    RuntimeArgConfig const &runtime_arg_config) {
  
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates> per_device_op_states;
  
  std::vector<parallel_layer_guid_t> pcg_layers = topological_ordering(pcg);
  
  // Process each layer in the PCG
  for (parallel_layer_guid_t const &parallel_layer : pcg_layers) {
    try {
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
      
      if (registry_contains_task_for_layer(backing->task_registry, regular_layer, OpTaskType::INIT)) {
        ParallelLayerAttrs parallel_layer_attrs = get_parallel_layer_attrs(pcg, parallel_layer);
        
        LayerAttrs layer_attrs = LayerAttrs{
          compgraph_op_attrs_from_pcg_op_attrs(parallel_layer_attrs.op_attrs),
          parallel_layer_attrs.name
        };
        
        std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(pcg, parallel_layer);
        std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(pcg, parallel_layer);
        std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(pcg, parallel_layer);
        
        std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
        std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
        std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
        
        std::vector<TensorShape> input_shapes;
        for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
          ParallelTensorShape parallel_shape = get_parallel_tensor_shape(pcg, parallel_tensor);
          input_shapes.push_back(get_piece_shape(parallel_shape));
        }
        
        // Create initialization task invocation
        TaskInvocation invocation = lower_to_task_invocation(
            init(layer_attrs.op_attrs), regular_layer, inputs, input_shapes, outputs, weights,
            backing->get_device_tensor_backing(device_id_t(gpu_id_t(nonnegative_int(0)))).tensor_gradient_mapping, std::nullopt);
        
        // Execute initialization on all available devices
        std::vector<Future<DeviceSpecificDeviceStates>> init_futures;
        size_t num_devices = std::min(backing->worker_procs.size(), backing->allocators.size());
        
        for (size_t i = 0; i < num_devices; i++) {
          device_id_t device = device_id_t(gpu_id_t(nonnegative_int(i)));
          Processor device_proc = backing->worker_procs[i];
          
          TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
              backing->get_device_tensor_backing(device),  // Use device-specific backing
              make_args_backing_with_empty_device_states(runtime_arg_config),
              invocation,
              device,
              *backing);
          
          task_id_t task_id = invocation.task_id;
          TaskImplFunction impl_function = backing->task_registry.task_mapping.at(task_id).impl_function;
          
          Promise<DeviceSpecificDeviceStates> promise(backing->master_mem);
          Future<DeviceSpecificDeviceStates> future = promise.get_future();
          RealmTaskArgs<DeviceSpecificDeviceStates>* task_arg = new RealmTaskArgs<DeviceSpecificDeviceStates>{
              task_id, impl_function, accessor, std::move(promise)};
          uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
          
          Event e = device_proc.spawn(get_realm_task_id(task_id), args, sizeof(uintptr_t), backing->worker_events[i]);
          backing->worker_events[i] = e;
          future.set_event(e);
          init_futures.push_back(future);
        }
        
        // Wait for all devices to complete initialization and combine results
        if (!init_futures.empty()) {
          try {
            std::vector<DeviceSpecificDeviceStates> device_states;
            device_states.reserve(init_futures.size());
            
            for (Future<DeviceSpecificDeviceStates> &future : init_futures) {
              device_states.push_back(future.get().value());
            }
            
            DeviceSpecificDeviceStates combined_state = combine_device_specific_states(device_states);
            
            per_device_op_states.insert({regular_layer, combined_state});
            
          } catch (const std::exception& e) {

            continue;
          }
        }
      }
    } catch (std::runtime_error const &e) {
      continue;
    }
  }
  
  return RealmArgsBacking{runtime_arg_config, per_device_op_states};
}

Future<float> execute_forward_pcg(RealmTrainingBackingPCG &backing,
                                 parallel_layer_guid_t const &layer) {
  
  std::vector<device_id_t> devices = get_layer_devices(backing, layer);
  
  if (devices.empty()) {
    return Future<float>(0.0f);
  }
  
  // Get layer attributes from PCG
  ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(backing.pcg, layer);
  PCGOperatorAttrs op_attrs = pcg_get_op_attrs(backing.pcg, layer);
  
  // Get parallelization strategy for this layer
  ParallelizationStrategy strategy = get_parallelization_strategy(backing.pcg, layer);
  
  // For data parallel, distribute batch across devices
  std::vector<TensorShape> device_input_shapes;
  if (strategy.type == ParallelizationType::DATA_PARALLEL) {
    device_input_shapes = create_data_parallel_input_shapes(backing, layer, devices);
  }
  
  std::vector<Future<float>> device_futures;
  device_futures.reserve(devices.size());
  
  // Create parallel execution contexts for all devices
  std::vector<std::unique_ptr<ParallelExecutionContext>> execution_contexts;
  
  for (size_t i = 0; i < devices.size(); i++) {
    device_id_t device = devices[i];
    
    auto context = std::make_unique<ParallelExecutionContext>(
        backing, layer, device, op_attrs);
    
    if (strategy.type == ParallelizationType::DATA_PARALLEL && 
        !device_input_shapes.empty() && i < device_input_shapes.size()) {

      context->device_input_shape = device_input_shapes[i];
    }
    
    Future<float> device_future = spawn_device_task_async(std::move(context));
    device_futures.push_back(device_future);
  }
  
  return combine_device_results_parallel(device_futures);
}

// Helper: Asynchronous task spawning for parallel execution
Future<float> spawn_device_task_async(std::unique_ptr<ParallelExecutionContext> context) {
  Processor device_proc = get_device_processor(context->backing, context->device);
  
  std::optional<TensorShape> device_input_shape = 
      context->device_input_shape.has_value() ? 
      std::optional<TensorShape>(context->device_input_shape) : std::nullopt;
  
  TaskInvocation invocation = create_task_invocation_for_device(
      context->backing, context->layer, context->device, context->op_attrs, device_input_shape);
  
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      context->backing.get_device_tensor_backing(context->device),  // Use device-specific backing
      context->backing.realm_args_backing,
      invocation,
      context->device,
      context->backing);
  
  // Create promise/future for result
  Promise<float> promise(context->backing.master_mem);
  Future<float> future = promise.get_future();
  
  RealmTaskArgs<float>* task_arg = new RealmTaskArgs<float>{
      invocation.task_id,
      context->backing.task_registry.task_mapping.at(invocation.task_id).impl_function,
      accessor,
      std::move(promise)
  };
  
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  Event spawn_event = device_proc.spawn(
      get_realm_task_id(invocation.task_id),
      args,
      sizeof(uintptr_t),
      Event::NO_EVENT  
  );
  
  future.set_event(spawn_event);
  return future;
}

Future<float> execute_forward_on_device(RealmTrainingBackingPCG &backing,
                                       parallel_layer_guid_t const &layer,
                                       device_id_t device,
                                       PCGOperatorAttrs const &attrs) {
  
  Processor device_proc = get_device_processor(backing, device);
  
  OpTaskInvocation op_invocation = forward(attrs);
  
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
  
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  
  std::optional<DeviceSpecificDeviceStates> device_state = 
      get_per_device_op_state_if_exists(backing.realm_args_backing, regular_layer);
  
  // Convert OpTaskInvocation to TaskInvocation
  TaskInvocation invocation = lower_to_task_invocation(
      op_invocation,
      regular_layer,
      inputs,
      input_shapes,
      outputs,
      weights,
      backing.get_device_tensor_backing(device).tensor_gradient_mapping,  // Use device-specific backing
      device_state);
  
  // Execute on the specific device
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.get_device_tensor_backing(device),  // Use device-specific backing
      backing.realm_args_backing,
      invocation,
      device,
      backing);
  
  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      backing.task_registry.task_mapping.at(task_id).impl_function;
  
  Promise<float> promise(backing.master_mem);
  Future<float> future = promise.get_future();
  RealmTaskArgs<float>* task_arg = new RealmTaskArgs<float>{
      task_id, impl_function, accessor, std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  Event e = device_proc.spawn(get_realm_task_id(task_id), args, sizeof(uintptr_t), Event::NO_EVENT);
  future.set_event(e);
  return future;
}

Future<float> execute_backward_pcg(RealmTrainingBackingPCG &backing,
                                  parallel_layer_guid_t const &layer,
                                  OptimizerAttrs const &optimizer_attrs) {  // ← Accept optimizer_attrs as parameter
  
  std::vector<device_id_t> devices = get_layer_devices(backing, layer);
  
  if (devices.empty()) {
    return Future<float>(0.0f);
  }
  
  PCGOperatorAttrs op_attrs = pcg_get_op_attrs(backing.pcg, layer);
  
  ParallelizationStrategy strategy = get_parallelization_strategy(backing.pcg, layer);
  
  // Execute on each device and combine results
  std::vector<Future<float>> device_futures;
  for (device_id_t device : devices) {
    Future<float> device_future = execute_backward_on_device(backing, layer, device, op_attrs);
    device_futures.push_back(device_future);
  }
  
  if (strategy.type == ParallelizationType::DATA_PARALLEL) {
    Future<float> backward_result = combine_device_results(device_futures);

    Future<void> sync_future = synchronize_gradients_data_parallel(backing, layer, devices, optimizer_attrs);  // ← Pass optimizer_attrs
    
    return backward_result;
  }
  
  return combine_device_results(device_futures);
}

Future<float> execute_backward_on_device(RealmTrainingBackingPCG &backing,
                                        parallel_layer_guid_t const &layer,
                                        device_id_t device,
                                        PCGOperatorAttrs const &attrs) {
  
  Processor device_proc = get_device_processor(backing, device);
  
  OpTaskInvocation op_invocation = backward(attrs);
  
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
  
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  
  std::optional<DeviceSpecificDeviceStates> device_state = 
      get_per_device_op_state_if_exists(backing.realm_args_backing, regular_layer);
  
  TaskInvocation invocation = lower_to_task_invocation(
      op_invocation,
      regular_layer,
      inputs,
      input_shapes,
      outputs,
      weights,
      backing.get_device_tensor_backing(device).tensor_gradient_mapping,  // Use device-specific backing
      device_state);
  
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.get_device_tensor_backing(device),  // Use device-specific backing
      backing.realm_args_backing,
      invocation,
      device,
      backing);
  
  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      backing.task_registry.task_mapping.at(task_id).impl_function;
  
  Promise<float> promise(backing.master_mem);
  Future<float> future = promise.get_future();
  RealmTaskArgs<float>* task_arg = new RealmTaskArgs<float>{
      task_id, impl_function, accessor, std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  Event e = device_proc.spawn(get_realm_task_id(task_id), args, sizeof(uintptr_t), Event::NO_EVENT);
  future.set_event(e);
  return future;
}

Future<void> execute_update_pcg(RealmTrainingBackingPCG &backing,
                               parallel_layer_guid_t const &layer,
                               OptimizerAttrs const &optimizer_attrs) {
  
  std::vector<device_id_t> devices = get_layer_devices(backing, layer);
  
  std::vector<Future<void>> update_futures;
  update_futures.reserve(devices.size());
  
  for (device_id_t device : devices) {
    Future<void> update_future = execute_update_on_device(backing, layer, device, optimizer_attrs);
    update_futures.push_back(update_future);
  }
  
  return combine_update_futures(update_futures);
}

Future<void> execute_update_on_device(RealmTrainingBackingPCG &backing,
                                     parallel_layer_guid_t const &layer,
                                     device_id_t device,
                                     OptimizerAttrs const &optimizer_attrs) {
  
  Processor device_proc = get_device_processor(backing, device);
  
  OpTaskInvocation op_invocation = update(optimizer_attrs);
  
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);

  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);

  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  
  std::optional<DeviceSpecificDeviceStates> device_state = 
      get_per_device_op_state_if_exists(backing.realm_args_backing, regular_layer);
  
  // Convert OpTaskInvocation to TaskInvocation
  TaskInvocation invocation = lower_to_task_invocation(
      op_invocation,
      regular_layer,
      inputs,
      input_shapes,
      outputs,
      weights,
      backing.get_device_tensor_backing(device).tensor_gradient_mapping,  // Use device-specific backing
      device_state);
  
  // Execute on the specific device
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.get_device_tensor_backing(device),  // Use device-specific backing
      backing.realm_args_backing,
      invocation,
      device,
      backing);
  
  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      backing.task_registry.task_mapping.at(task_id).impl_function;
  
  Promise<void> promise(backing.master_mem);
  Future<void> future = promise.get_future();
  RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{
      task_id, impl_function, accessor, std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  Event e = device_proc.spawn(get_realm_task_id(task_id), args, sizeof(uintptr_t), Event::NO_EVENT);
  future.set_event(e);
  return future;
}

Future<void> compute_loss_pcg(RealmTrainingBackingPCG &backing,
                             LossAttrs const &loss_attrs,
                             parallel_tensor_guid_t const &logit_tensor,
                             loss_tensor_t const &label_tensor) {
  
  std::vector<device_id_t> devices = get_tensor_devices(backing, logit_tensor);
  
  std::vector<Future<void>> loss_futures;
  loss_futures.reserve(devices.size());
  
  for (device_id_t device : devices) {
    Future<void> loss_future = compute_loss_on_device(backing, loss_attrs, logit_tensor, label_tensor, device);
    loss_futures.push_back(loss_future);
  }
  
  return combine_loss_futures(loss_futures);
}

Future<void> compute_loss_on_device(RealmTrainingBackingPCG &backing,
                                   LossAttrs const &loss_attrs,
                                   parallel_tensor_guid_t const &logit_tensor,
                                   loss_tensor_t const &label_tensor,
                                   device_id_t device) {
  
  // Get device-specific processor
  Processor device_proc = get_device_processor(backing, device);
  
  OpTaskInvocation op_invocation = compute_loss(loss_attrs);
  
  tensor_guid_t regular_logit_tensor = convert_parallel_to_regular_tensor(logit_tensor);
  
  // Create task invocation for loss computation
  TaskInvocation invocation = lower_to_task_invocation(
      op_invocation,
      layer_guid_t{}, // Loss doesn't have a specific layer
      {regular_logit_tensor}, // logit tensor
      {}, // No input shapes needed for loss
      {}, // No outputs for loss computation
      {}, // No weights for loss
      backing.get_device_tensor_backing(device).tensor_gradient_mapping,  // Use device-specific backing
      std::nullopt);
  
  // Execute on the specific device
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.get_device_tensor_backing(device),  
      backing.realm_args_backing,
      invocation,
      device,
      backing);
  
  task_id_t task_id = invocation.task_id;
  TaskImplFunction impl_function =
      backing.task_registry.task_mapping.at(task_id).impl_function;
  
  Promise<void> promise(backing.master_mem);
  Future<void> future = promise.get_future();
  RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{
      task_id, impl_function, accessor, std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  Event e = device_proc.spawn(get_realm_task_id(task_id), args, sizeof(uintptr_t), Event::NO_EVENT);
  future.set_event(e);
  return future;
}

// Device management functions
std::vector<device_id_t> get_layer_devices(RealmTrainingBackingPCG const &backing,
                                          parallel_layer_guid_t const &layer) {
  std::vector<device_id_t> devices;
  
  try {
    UnstructuredDeviceMapping device_mapping = 
        get_unstructured_device_mapping(backing.machine_mapping, backing.machine_spec, backing.pcg);
    
    ParallelizationStrategy strategy = get_parallelization_strategy(backing.pcg, layer);
    
    std::vector<device_id_t> pcg_devices = get_layer_device_placement(device_mapping, layer);
    
    for (device_id_t device : pcg_devices) {
      if (is_device_available(backing, device)) {
        devices.push_back(device);
      }
    }
    
    if (devices.empty()) {
      devices = get_devices_by_strategy(backing, layer, strategy);
    }
    
  } catch (const std::exception& e) {
    // Fallback to basic device assignment if PCG mapping fails
    devices = get_fallback_devices(backing, layer);
  }

  if (devices.empty()) {
    devices.push_back(device_id_t(gpu_id_t(nonnegative_int(0))));
  }
  
  return devices;
}

// Helper: Get devices based on parallelization strategy
std::vector<device_id_t> get_devices_by_strategy(
    RealmTrainingBackingPCG const &backing,
    parallel_layer_guid_t const &layer,
    ParallelizationStrategy strategy) {
  
  std::vector<device_id_t> devices;
  size_t available_devices = std::min(backing.worker_procs.size(), backing.allocators.size());
  
  switch (strategy.type) {
    case ParallelizationType::DATA_PARALLEL:
      for (size_t i = 0; i < available_devices; i++) {
        devices.push_back(device_id_t(gpu_id_t(nonnegative_int(i))));
      }
      break;
      
    case ParallelizationType::MODEL_PARALLEL:
      {
        size_t partition_size = strategy.partition_size;
        size_t num_partitions = std::min(available_devices, partition_size);
        for (size_t i = 0; i < num_partitions; i++) {
          devices.push_back(device_id_t(gpu_id_t(nonnegative_int(i))));
        }
      }
      break;
      
    case ParallelizationType::PIPELINE_PARALLEL:
      {
        size_t stage_id = strategy.stage_id;
        if (stage_id < available_devices) {
          devices.push_back(device_id_t(gpu_id_t(nonnegative_int(stage_id))));
        }
      }
      break;
      
    default:
      devices.push_back(device_id_t(gpu_id_t(nonnegative_int(0))));
      break;
  }
  
  return devices;
}

// Helper: Check if device is available in backing
bool is_device_available(RealmTrainingBackingPCG const &backing, device_id_t device) {
  auto gpu_index = device.gpu_id.gpu_index.raw_value;
  return gpu_index < backing.worker_procs.size() && 
         gpu_index < backing.allocators.size();
}

// Helper: Fallback device assignment
std::vector<device_id_t> get_fallback_devices(
    RealmTrainingBackingPCG const &backing,
    parallel_layer_guid_t const &layer) {
  
  std::vector<device_id_t> devices;
  size_t num_devices = std::min(backing.worker_procs.size(), backing.allocators.size());
  
  // Use all available devices for maximum parallelism
  for (size_t i = 0; i < num_devices; i++) {
    devices.push_back(device_id_t(gpu_id_t(nonnegative_int(i))));
  }
  
  return devices;
}

// Helper: Get parallelization strategy from PCG
ParallelizationStrategy get_parallelization_strategy(
    ParallelComputationGraph const &pcg,
    parallel_layer_guid_t const &layer) {
  
  try {
    ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(pcg, layer);
    
    PCGOperatorAttrs op_attrs = layer_attrs.op_attrs;
    
    return infer_parallelization_strategy(op_attrs);
    
  } catch (const std::exception& e) {
    // Default to data parallelism 
    return ParallelizationStrategy{
        .type = ParallelizationType::DATA_PARALLEL,
        .partition_size = 1,
        .stage_id = 0
    };
  }
}

// Helper: Infer parallelization strategy from operator attributes
// default to data parallelism regardless of operator attributes
ParallelizationStrategy infer_parallelization_strategy(PCGOperatorAttrs const &op_attrs) {
  return ParallelizationStrategy{
      .type = ParallelizationType::DATA_PARALLEL,
      .partition_size = 1,
      .stage_id = 0
  };
}

Processor get_device_processor(RealmTrainingBackingPCG const &backing,
                              device_id_t device_id) {
  auto it = backing.device_to_processor.find(device_id);
  if (it != backing.device_to_processor.end()) {
    return it->second;
  }
  // Fallback: return first processor
  return backing.worker_procs[0];
}

Allocator &get_device_allocator(RealmTrainingBackingPCG &backing,
                               device_id_t device_id) {

  auto gpu_id = device_id.gpu_id.gpu_index;
  size_t allocator_index = gpu_id.raw_value % backing.allocators.size();
  
  return const_cast<Allocator&>(backing.allocators[allocator_index]);
}

// Helper methods for device-specific tensor access
RealmTensorBacking const &RealmTrainingBackingPCG::get_device_tensor_backing(device_id_t device) const {
  auto it = device_tensor_backings.find(device);
  if (it == device_tensor_backings.end()) {
    throw std::runtime_error(fmt::format("No tensor backing found for device {}", device.gpu_id.gpu_index.raw_value));
  }
  return it->second;
}

RealmTensorBacking &RealmTrainingBackingPCG::get_device_tensor_backing(device_id_t device) {
  auto it = device_tensor_backings.find(device);
  if (it == device_tensor_backings.end()) {
    throw std::runtime_error(fmt::format("No tensor backing found for device {}", device.gpu_id.gpu_index.raw_value));
  }
  return it->second;
}

// Update function signatures to use device-specific tensor backings
TaskArgumentAccessor get_task_arg_accessor_pcg(
    RealmTensorBacking const &device_tensor_backing,
    RealmArgsBacking const &realm_args_backing,
    TaskInvocation const &invocation,
    device_id_t target_device,
    RealmTrainingBackingPCG &backing) {
  
  TensorSlotsBacking tensor_slots_backing =
      construct_tensor_slots_backing(device_tensor_backing, invocation.binding);
  ArgSlotsBacking arg_slots_backing = construct_arg_slots_backing(
      invocation.binding, realm_args_backing.runtime_arg_config);
      
  // Multi-GPU: use device-specific allocator
  Allocator &device_allocator = get_device_allocator(backing, target_device);
  return TaskArgumentAccessor::create<RealmTaskArgumentAccessor>(
      device_allocator, tensor_slots_backing, arg_slots_backing);
}

// Helper functions for multi-device result combination
Future<float> combine_device_results(std::vector<Future<float>> const &device_futures) {
  if (device_futures.empty()) {
    return Future<float>(0.0f);
  }
  
  if (device_futures.size() == 1) {
    return device_futures[0];
  }
  
  // Create a combined future that waits for all device futures
  Promise<float> combined_promise;
  Future<float> combined_future = combined_promise.get_future();
  
  auto combination_task = [device_futures, promise = std::move(combined_promise)]() mutable {
    try {
      std::vector<float> device_results;
      device_results.reserve(device_futures.size());
      
      for (Future<float> const &future : device_futures) {
        device_results.push_back(future.get());
      }
      
      float combined_result = combine_parallel_results(device_results);
      
      promise.set_value(combined_result);
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  std::thread(combination_task).detach();
  
  return combined_future;
}

// parallel result combination
Future<float> combine_device_results_parallel(std::vector<Future<float>> const &device_futures) {
  if (device_futures.empty()) {
    return Future<float>(0.0f);
  }
  
  if (device_futures.size() == 1) {
    return device_futures[0];
  }
  
  // Create a combined future that waits for all device futures
  Promise<float> combined_promise;
  Future<float> combined_future = combined_promise.get_future();
  
  auto combination_task = [device_futures, promise = std::move(combined_promise)]() mutable {
    try {
      std::vector<float> device_results;
      device_results.reserve(device_futures.size());
      

      for (Future<float> const &future : device_futures) {
        device_results.push_back(future.get());
      }
      
      float combined_result = combine_parallel_results(device_results);
      
      promise.set_value(combined_result);
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  

  std::thread(combination_task).detach();
  
  return combined_future;
}

// Helper: Combine results from multiple devices based on parallelization strategy
float combine_parallel_results(std::vector<float> const &device_results) {
  if (device_results.empty()) {
    return 0.0f;
  }
  
  // Data Parallelism - Average the results
  float sum = 0.0f;
  for (float result : device_results) {
    sum += result;
  }
  return sum / static_cast<float>(device_results.size());
}

Future<void> combine_update_futures(std::vector<Future<void>> const &update_futures) {
  if (update_futures.empty()) {
    return Future<void>();
  }
  
  if (update_futures.size() == 1) {
    return update_futures[0];
  }
  
  // Create a combined future that waits for all update operations
  Promise<void> combined_promise;
  Future<void> combined_future = combined_promise.get_future();
  
  auto combination_task = [update_futures, promise = std::move(combined_promise)]() mutable {
    try {
      for (Future<void> const &future : update_futures) {
        future.get();
      }
      promise.set_value();
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  std::thread(combination_task).detach();
  
  return combined_future;
}

Future<void> combine_loss_futures(std::vector<Future<void>> const &loss_futures) {
  if (loss_futures.empty()) {
    return Future<void>();
  }
  
  if (loss_futures.size() == 1) {
    return loss_futures[0];
  }
  
  Promise<void> combined_promise;
  Future<void> combined_future = combined_promise.get_future();
  
  auto combination_task = [loss_futures, promise = std::move(combined_promise)]() mutable {
    try {
      for (Future<void> const &future : loss_futures) {
        future.get();
      }
      promise.set_value();
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  std::thread(combination_task).detach();
  
  return combined_future;
}

// Helper: Combine device-specific states from multiple devices
DeviceSpecificDeviceStates combine_device_specific_states(
    std::vector<DeviceSpecificDeviceStates> const &device_states) {
  
  if (device_states.empty()) {
    return DeviceSpecificDeviceStates{};
  }
  
  if (device_states.size() == 1) {
    return device_states[0];
  }

  DeviceSpecificDeviceStates combined_state = device_states[0];
  
  for (size_t i = 1; i < device_states.size(); i++) {
    combined_state = combine_device_states_with_tolerance(
        combined_state, device_states[i]);
  }
  
  return combined_state;
}

// Helper: Combine two device states with tolerance for differences
DeviceSpecificDeviceStates combine_device_states_with_tolerance(
    DeviceSpecificDeviceStates const &state1,
    DeviceSpecificDeviceStates const &state2) {
  
  DeviceSpecificDeviceStates combined_state;
  
  // Combine per-layer states
  for (auto const &layer_pair : state1.per_layer_states) {
    layer_guid_t layer = layer_pair.first;
    PerDeviceOpState const &state1_layer = layer_pair.second;
    
    auto state2_it = state2.per_layer_states.find(layer);
    if (state2_it != state2.per_layer_states.end()) {
      PerDeviceOpState const &state2_layer = state2_it->second;
      
      PerDeviceOpState combined_layer_state = combine_layer_states_with_tolerance(
          state1_layer, state2_layer);
      
      combined_state.per_layer_states[layer] = combined_layer_state;
    } else {

      combined_state.per_layer_states[layer] = state1_layer;
    }
  }
  
  // Add layers that only exist in state2
  for (auto const &layer_pair : state2.per_layer_states) {
    layer_guid_t layer = layer_pair.first;
    if (combined_state.per_layer_states.find(layer) == combined_state.per_layer_states.end()) {
      combined_state.per_layer_states[layer] = layer_pair.second;
    }
  }
  
  return combined_state;
}

// Helper: Combine layer states with tolerance for floating-point differences
PerDeviceOpState combine_layer_states_with_tolerance(
    PerDeviceOpState const &state1,
    PerDeviceOpState const &state2) {
  
  PerDeviceOpState combined_state;
  
  // Combine handles (use first non-null handle)
  if (state1.handle.blas != nullptr) {
    combined_state.handle.blas = state1.handle.blas;
  } else if (state2.handle.blas != nullptr) {
    combined_state.handle.blas = state2.handle.blas;
  }
  
  if (state1.handle.dnn != nullptr) {
    combined_state.handle.dnn = state1.handle.dnn;
  } else if (state2.handle.dnn != nullptr) {
    combined_state.handle.dnn = state2.handle.dnn;
  }
  
  // Combine other state fields with tolerance
  // For numeric fields, use average or first non-zero value
  // For boolean fields, use logical OR
  // For pointer fields, use first non-null pointer
  
  // Example: combine activation states
  if (state1.activation != ActivationMode::NONE) {
    combined_state.activation = state1.activation;
  } else if (state2.activation != ActivationMode::NONE) {
    combined_state.activation = state2.activation;
  }
  
  // Example: combine dropout states
  if (state1.dropout_rate > 0.0f) {
    combined_state.dropout_rate = state1.dropout_rate;
  } else if (state2.dropout_rate > 0.0f) {
    combined_state.dropout_rate = state2.dropout_rate;
  }
  
  // TODO: other fields
  
  return combined_state;
}

// Helper: Compare floating-point values with tolerance
bool float_equal_with_tolerance(float a, float b, float tolerance = 1e-6f) {
  return std::abs(a - b) <= tolerance;
}

// Helper: Compare double values with tolerance
bool double_equal_with_tolerance(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}

// Helper: Combine numeric values with tolerance
float combine_float_values_with_tolerance(float a, float b, float tolerance = 1e-6f) {
  if (float_equal_with_tolerance(a, b, tolerance)) {
    return a;  // Values are effectively equal, use either
  } else {
    // Values are different, use average or first non-zero
    if (std::abs(a) > tolerance) {
      return a;
    } else if (std::abs(b) > tolerance) {
      return b;
    } else {
      return (a + b) / 2.0f;  
    }
  }
}

// Placeholder implementations for missing conversion functions
layer_guid_t convert_parallel_to_regular_layer(parallel_layer_guid_t const &parallel_layer) {
  return layer_guid_t{parallel_layer.raw_graph_node};
}

tensor_guid_t convert_parallel_to_regular_tensor(parallel_tensor_guid_t const &parallel_tensor) {
  return tensor_guid_t{parallel_tensor.raw_graph_output};
}

parallel_layer_guid_t convert_regular_to_parallel_layer(layer_guid_t const &regular_layer) {
  return parallel_layer_guid_t{regular_layer.raw_node};
}

parallel_tensor_guid_t convert_regular_to_parallel_tensor(tensor_guid_t const &regular_tensor) {
  return parallel_tensor_guid_t{regular_tensor.raw_graph_output};
}


// Helper: Distribute batch data across devices for data parallel execution
std::vector<TensorShape> distribute_batch_data_parallel(
    TensorShape const &original_shape,
    size_t num_devices) {
  
  std::vector<TensorShape> distributed_shapes;
  distributed_shapes.reserve(num_devices);
  
  size_t batch_size = original_shape.dims.back().size;
  size_t batch_per_device = batch_size / num_devices;
  
  if (batch_per_device == 0) {
    distributed_shapes.push_back(original_shape);
    return distributed_shapes;
  }
  
  for (size_t i = 0; i < num_devices; i++) {
    TensorShape device_shape = original_shape;
    
    if (i == num_devices - 1) {
      device_shape.dims.back().size = batch_size - (batch_per_device * (num_devices - 1));
    } else {
      device_shape.dims.back().size = batch_per_device;
    }
    
    distributed_shapes.push_back(device_shape);
  }
  
  return distributed_shapes;
}

// Helper: Create device-specific tensor shapes for data parallel execution
std::vector<TensorShape> create_data_parallel_input_shapes(
    RealmTrainingBackingPCG const &backing,
    parallel_layer_guid_t const &layer,
    std::vector<device_id_t> const &devices) {
  
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  
  if (parallel_inputs.empty()) {
    return {};
  }
  
  parallel_tensor_guid_t primary_input = parallel_inputs[0];
  ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, primary_input);
  TensorShape original_shape = get_piece_shape(parallel_shape);
  
  return distribute_batch_data_parallel(original_shape, devices.size());
}

// Helper: Synchronize gradients across devices for data parallel training
Future<void> synchronize_gradients_data_parallel(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    std::vector<device_id_t> const &devices,
    OptimizerAttrs const &optimizer_attrs) {  
  
  // TODO: All-reduce
  
  std::vector<Future<void>> sync_futures;
  sync_futures.reserve(devices.size());
  
  for (device_id_t device : devices) {
    // Create gradient synchronization task for this device
    Future<void> sync_future = synchronize_gradients_on_device(backing, layer, device, optimizer_attrs);  // ← Pass optimizer_attrs
    sync_futures.push_back(sync_future);
  }
  
  return combine_sync_futures(sync_futures);
}

// Helper: Synchronize gradients on a specific device
Future<void> synchronize_gradients_on_device(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    device_id_t device,
    OptimizerAttrs const &optimizer_attrs) {  
  
  Processor device_proc = get_device_processor(backing, device);
  
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  // All-reduce
  Promise<void> promise(backing.master_mem);
  Future<void> future = promise.get_future();
  
  // For each weight tensor, perform all-reduce on its gradients
  std::vector<Future<void>> weight_sync_futures;
  weight_sync_futures.reserve(weights.size());
  
  for (tensor_guid_t const &weight : weights) {
    auto grad_it = backing.get_device_tensor_backing(device).tensor_gradient_mapping.find(weight);
    if (grad_it != backing.get_device_tensor_backing(device).tensor_gradient_mapping.end()) {
      Future<void> weight_sync = perform_all_reduce_on_device(
          backing, weight, grad_it->second, device, device_proc, optimizer_attrs);
      weight_sync_futures.push_back(weight_sync);
    }
  }
  
  if (!weight_sync_futures.empty()) {
    auto combined_future = combine_weight_sync_futures(weight_sync_futures);
    combined_future.then([promise = std::move(promise)]() mutable {
      promise.set_value();
    });
  } else {
    promise.set_value();
  }
  
  return future;
}

// Helper: Perform all-reduce on a specific weight's gradients
Future<void> perform_all_reduce_on_device(
    RealmTrainingBackingPCG &backing,
    tensor_guid_t const &weight,
    tensor_guid_t const &gradient,
    device_id_t device,
    Processor device_proc,
    OptimizerAttrs const &optimizer_attrs) {  
  

  Promise<void> promise(backing.master_mem);
  Future<void> future = promise.get_future();
  
  std::vector<optimizer_tensor_t> optimizer_buffer_tensors;
  auto opt_it = backing.get_device_tensor_backing(device).tensor_optimizer_mapping.find(weight);
  if (opt_it != backing.get_device_tensor_backing(device).tensor_optimizer_mapping.end()) {
    optimizer_buffer_tensors = opt_it->second;
  }
  
  TaskInvocation update_invocation = get_update_invocation(
      optimizer_attrs, weight, gradient, optimizer_buffer_tensors);
  
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.get_device_tensor_backing(device),  
      backing.realm_args_backing,
      update_invocation,
      device,
      backing);
  
  task_id_t task_id = update_invocation.task_id;
  TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
  
  // Create task arguments
  RealmTaskArgs<void>* task_arg = new RealmTaskArgs<void>{
      task_id, update_impl_fn, accessor, std::move(promise)};
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  

  Event e = device_proc.spawn(get_realm_task_id(task_id), args, sizeof(uintptr_t), Event::NO_EVENT);
  future.set_event(e);
  
  return future;
}

// Helper: Combine weight synchronization futures
Future<void> combine_weight_sync_futures(std::vector<Future<void>> const &weight_sync_futures) {
  if (weight_sync_futures.empty()) {
    return Future<void>();
  }
  
  if (weight_sync_futures.size() == 1) {
    return weight_sync_futures[0];
  }
  
  // Create a combined future that waits for all weight sync operations
  Promise<void> combined_promise;
  Future<void> combined_future = combined_promise.get_future();
  
  auto combination_task = [weight_sync_futures, promise = std::move(combined_promise)]() mutable {
    try {
      // Wait for all weight synchronization operations to complete
      for (Future<void> const &future : weight_sync_futures) {
        future.get();
      }
      promise.set_value();
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  std::thread(combination_task).detach();
  
  return combined_future;
}

// Helper: Combine synchronization futures
Future<void> combine_sync_futures(std::vector<Future<void>> const &sync_futures) {
  if (sync_futures.empty()) {
    return Future<void>();
  }
  
  if (sync_futures.size() == 1) {
    return sync_futures[0];
  }
  
  // Create a combined future that waits for all synchronization operations
  Promise<void> combined_promise;
  Future<void> combined_future = combined_promise.get_future();
  
  auto combination_task = [sync_futures, promise = std::move(combined_promise)]() mutable {
    try {
      // Wait for all synchronization operations to complete
      for (Future<void> const &future : sync_futures) {
        future.get();
      }
      promise.set_value();
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  std::thread(combination_task).detach();
  
  return combined_future;
}

// Helper: Synchronize device states across all devices
Future<void> synchronize_device_states(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    std::vector<device_id_t> const &devices) {
  
  std::vector<Future<DeviceSpecificDeviceStates>> device_state_futures;
  device_state_futures.reserve(devices.size());
  
  for (device_id_t device : devices) {
    layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
    
    // Create a future that will be resolved with the device state
    Promise<DeviceSpecificDeviceStates> promise(backing.master_mem);
    Future<DeviceSpecificDeviceStates> future = promise.get_future();
    
    // In a real implementation, this would query the actual device state
    // For now, we'll create a placeholder that represents the device state
    DeviceSpecificDeviceStates device_state = get_device_state_for_layer(
        backing, regular_layer, device);
    
    promise.set_value(device_state);
    device_state_futures.push_back(future);
  }
  
  // Wait for all device states and combine them
  Promise<void> sync_promise(backing.master_mem);
  Future<void> sync_future = sync_promise.get_future();
  
  auto sync_task = [device_state_futures, &backing, layer, promise = std::move(sync_promise)]() mutable {
    try {
      std::vector<DeviceSpecificDeviceStates> device_states;
      device_states.reserve(device_state_futures.size());
      
      // Collect all device states
      for (Future<DeviceSpecificDeviceStates> &future : device_state_futures) {
        device_states.push_back(future.get());
      }
      
      DeviceSpecificDeviceStates combined_state = combine_device_specific_states(device_states);
      
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
      store_combined_device_state(backing, regular_layer, combined_state);
      
      promise.set_value();
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  std::thread(sync_task).detach();
  return sync_future;
}

// Helper: Get device state for a specific layer and device
DeviceSpecificDeviceStates get_device_state_for_layer(
    RealmTrainingBackingPCG &backing,
    layer_guid_t const &layer,
    device_id_t device) {
  

  DeviceSpecificDeviceStates device_state;
  
  auto it = backing.realm_args_backing.per_device_op_states.find(layer);
  if (it != backing.realm_args_backing.per_device_op_states.end()) {
    device_state.per_layer_states[layer] = it->second;
  }
  
  return device_state;
}

// Helper: Store combined device state
void store_combined_device_state(
    RealmTrainingBackingPCG &backing,
    layer_guid_t const &layer,
    DeviceSpecificDeviceStates const &combined_state) {
  
  // TODO
}

// PCG integration functions using actual PCG API
std::unordered_map<layer_guid_t, LayerAttrs> get_layer_attrs_mapping_from_pcg(ParallelComputationGraph const &pcg) {
  std::unordered_map<layer_guid_t, LayerAttrs> layer_attrs_mapping;
  
  std::unordered_set<parallel_layer_guid_t> parallel_layers = get_parallel_layers(pcg);
  
  for (parallel_layer_guid_t const &parallel_layer : parallel_layers) {
    try {
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
      
      ParallelLayerAttrs parallel_attrs = get_parallel_layer_attrs(pcg, parallel_layer);
      
      LayerAttrs layer_attrs = LayerAttrs{
        compgraph_op_attrs_from_pcg_op_attrs(parallel_attrs.op_attrs),
        parallel_attrs.name
      };
      
      layer_attrs_mapping[regular_layer] = layer_attrs;
    } catch (std::runtime_error const &e) {
      continue;
    }
  }
  
  return layer_attrs_mapping;
}

std::unordered_map<tensor_guid_t, TensorAttrs> get_all_tensor_attrs_from_pcg(ParallelComputationGraph const &pcg) {
  std::unordered_map<tensor_guid_t, TensorAttrs> tensor_attrs_mapping;
  
  std::unordered_set<parallel_tensor_guid_t> parallel_tensors = get_parallel_tensors(pcg);
  
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_tensors) {
    try {
      tensor_guid_t regular_tensor = convert_parallel_to_regular_tensor(parallel_tensor);

      ParallelTensorAttrs parallel_attrs = get_parallel_tensor_attrs(pcg, parallel_tensor);
      
      TensorAttrs tensor_attrs = get_piece_attrs(parallel_attrs);
      
      tensor_attrs_mapping[regular_tensor] = tensor_attrs;
    } catch (std::runtime_error const &e) {
      continue;
    }
  }
  
  return tensor_attrs_mapping;
}

LayerAttrs get_layer_attrs_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  
  ParallelLayerAttrs parallel_attrs = get_parallel_layer_attrs(pcg, parallel_layer);

  return LayerAttrs{
    compgraph_op_attrs_from_pcg_op_attrs(parallel_attrs.op_attrs),
    parallel_attrs.name
  };
}

std::vector<layer_guid_t> topological_ordering_from_pcg(ParallelComputationGraph const &pcg) {
  std::vector<parallel_layer_guid_t> parallel_ordering = topological_ordering(pcg);
  std::vector<layer_guid_t> regular_ordering;
  
  for (parallel_layer_guid_t const &parallel_layer : parallel_ordering) {
    try {
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
      regular_ordering.push_back(regular_layer);
    } catch (std::runtime_error const &e) {
      continue;
    }
  }
  
  return regular_ordering;
}

std::vector<tensor_guid_t> get_incoming_inputs_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(pcg, parallel_layer);
  
  std::vector<tensor_guid_t> regular_inputs;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    regular_inputs.push_back(convert_parallel_to_regular_tensor(parallel_tensor));
  }
  return regular_inputs;
}

std::vector<TensorShape> get_incoming_input_shapes_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(pcg, parallel_layer);
  
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  return input_shapes;
}

std::vector<tensor_guid_t> get_outgoing_tensors_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(pcg, parallel_layer);
  
  std::vector<tensor_guid_t> regular_outputs;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_outputs) {
    regular_outputs.push_back(convert_parallel_to_regular_tensor(parallel_tensor));
  }
  return regular_outputs;
}

std::vector<tensor_guid_t> get_incoming_weights_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(pcg, parallel_layer);

  std::vector<tensor_guid_t> regular_weights;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_weights) {
    regular_weights.push_back(convert_parallel_to_regular_tensor(parallel_tensor));
  }
  return regular_weights;
}

std::vector<device_id_t> get_tensor_devices(RealmTrainingBackingPCG const &backing, parallel_tensor_guid_t const &tensor) {
  parallel_layer_guid_t source_layer = get_source_layer(backing.pcg, tensor);
  return get_layer_devices(backing, source_layer);
}

// Helper: Physically replicate tensors for a specific device
AllocatedTensors replicate_tensors_for_device(
    AllocatedTensors const &source_tensors,
    device_id_t device,
    Allocator &device_allocator) {
  
  AllocatedTensors device_tensors;
  
  for (auto const &tensor_pair : source_tensors) {
    tensor_guid_t tensor_guid = tensor_pair.first;
    GenericTensorAccessorW source_accessor = tensor_pair.second;
    
    TensorShape tensor_shape = source_accessor.domain;
    DataType data_type = source_accessor.data_type;
    
    GenericTensorAccessorW device_accessor = 
        allocate_tensor_on_device(tensor_shape, data_type, device_allocator);
    
    //  Copy actual tensor values from source to device
    copy_tensor_values(source_accessor, device_accessor);
    
    device_tensors[tensor_guid] = device_accessor;
  }
  
  return device_tensors;
}

// Helper: Physically replicate unallocated tensors for a specific device
UnallocatedTensors replicate_unallocated_tensors_for_device(
    UnallocatedTensors const &source_tensors,
    device_id_t device,
    Allocator &device_allocator) {
  
  UnallocatedTensors device_tensors;
  
  for (auto const &tensor_pair : source_tensors) {
    tensor_guid_t tensor_guid = tensor_pair.first;
    TensorAttrs tensor_attrs = tensor_pair.second;
    
    // Create device-specific tensor attributes
    device_tensors[tensor_guid] = tensor_attrs;
  }
  
  return device_tensors;
}

// Helper: Calculate tensor size in bytes
size_t calculate_tensor_size(TensorShape const &shape, DataType data_type) {
  size_t num_elements = 1;
  for (auto const &dim : shape.dims) {
    num_elements *= dim.size;
  }
  
  size_t element_size = get_element_size(data_type);
  return num_elements * element_size;
}

// Helper: Create tensor accessor for device-specific memory
GenericTensorAccessorW create_tensor_accessor(
    void* device_memory,
    TensorShape const &shape,
    DataType data_type) {
  
  // Create domain from shape
  Domain domain;
  for (auto const &dim : shape.dims) {
    domain.add_dim(dim.size);
  }
  
  // Create device-specific tensor accessor
  return GenericTensorAccessorW(device_memory, domain, data_type);
}

// Helper: Allocate tensor on specific device
GenericTensorAccessorW allocate_tensor_on_device(
    TensorShape const &shape,
    DataType data_type,
    Allocator &device_allocator) {
  
  // Calculate tensor size
  size_t tensor_size = calculate_tensor_size(shape, data_type);
  
  // Allocate memory on this specific device
  void* device_memory = device_allocator.allocate(tensor_size);
  
  // Create device-specific accessor
  return create_tensor_accessor(device_memory, shape, data_type);
}

// Helper: Copy tensor values from source to destination accessor
void copy_tensor_values(GenericTensorAccessorW const &source_accessor,
                       GenericTensorAccessorW &dest_accessor) {
  
  if (source_accessor.domain != dest_accessor.domain) {
    throw std::runtime_error("Tensor shapes must match for copying");
  }
  
  if (source_accessor.data_type != dest_accessor.data_type) {
    throw std::runtime_error("Tensor data types must match for copying");
  }
  
  if (source_accessor.ptr == nullptr) {
    throw std::runtime_error("Source tensor pointer is null");
  }
  
  if (dest_accessor.ptr == nullptr) {
    throw std::runtime_error("Destination tensor pointer is null");
  }
  
  size_t num_elements = 1;
  for (auto const &dim : source_accessor.domain.dims) {
    num_elements *= dim.size;
  }
  
  size_t element_size = get_element_size(source_accessor.data_type);
  size_t total_bytes = num_elements * element_size;
  
  // Copy data from source to destination
  void* source_ptr = source_accessor.ptr;
  void* dest_ptr = dest_accessor.ptr;
  
  // NOTE: This will not work for GPU-to-GPU transfers (TODO)
  std::memcpy(dest_ptr, source_ptr, total_bytes);
  
}

// Helper: Get element size in bytes for a data type
size_t get_element_size(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT64:
      return sizeof(double);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::BOOL:
      return sizeof(bool);
    case DataType::INT8:
      return sizeof(int8_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::INT16:
      return sizeof(int16_t);
    case DataType::UINT16:
      return sizeof(uint16_t);
    case DataType::UINT32:
      return sizeof(uint32_t);
    case DataType::UINT64:
      return sizeof(uint64_t);
    default:
      throw std::runtime_error("Unsupported data type for tensor copying");
  }
}

} // namespace FlexFlow
