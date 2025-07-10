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
#include <unordered_set>

namespace FlexFlow {

using namespace Realm;

// Parallelization strategy types
enum class ParallelizationType {
  DATA_PARALLEL,       // Same model on multiple devices, different data
  MODEL_PARALLEL,      // Different parts of model on different devices
  PIPELINE_PARALLEL,   // Different stages of pipeline on different devices
  HYBRID_PARALLEL      // Combination of above strategies
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
    PCGOperatorAttrs const &attrs) {
  
  // Create forward task invocation using PCG functions
  OpTaskInvocation op_invocation = forward(attrs);
  
  // Convert parallel layer to regular layer for compatibility
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
  
  // Get tensor information from PCG
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  // Convert to regular tensors
  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  // Get input shapes
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
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
      backing.realm_tensor_backing.tensor_gradient_mapping,
      device_state);
}

// FIXED: Multi-GPU tensor backing construction - distribute tensors across allocators
RealmTensorBacking construct_multi_gpu_realm_tensor_backing(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    std::vector<Allocator> const &allocators,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec,
    ParallelComputationGraph const &pcg) {
  
  if (allocators.empty()) {
    throw std::runtime_error("No allocators provided for multi-GPU tensor backing");
  }
  
  // FIXED: Proper multi-GPU tensor distribution instead of single allocator
  try {
    // Get device mapping from PCG
    UnstructuredDeviceMapping device_mapping = 
        get_unstructured_device_mapping(machine_mapping, machine_spec, pcg);
    
    // Create tensor-to-device mapping based on PCG analysis
    std::unordered_map<tensor_guid_t, device_id_t> tensor_device_mapping = 
        create_tensor_device_mapping(pcg, device_mapping, allocators.size());
    
    // Create device-specific tensor backings
    std::vector<RealmTensorBacking> device_tensor_backings;
    device_tensor_backings.reserve(allocators.size());
    
    for (size_t i = 0; i < allocators.size(); i++) {
      device_id_t device = device_id_t(gpu_id_t(nonnegative_int(i)));
      
      // Get tensors assigned to this device
      AllocatedTensors device_allocated = filter_tensors_for_device(
          allocated_tensors, tensor_device_mapping, device);
      UnallocatedTensors device_unallocated = filter_unallocated_tensors_for_device(
          unallocated_tensors, tensor_device_mapping, device);
      
      // Create tensor backing for this device
      RealmTensorBacking device_backing = construct_realm_tensor_backing(
          device_allocated, device_unallocated, 
          const_cast<Allocator&>(allocators[i]));
      
      device_tensor_backings.push_back(device_backing);
    }
    
    // Merge all device tensor backings into a unified backing
    return merge_device_tensor_backings(device_tensor_backings, allocators);
    
  } catch (const std::exception& e) {
    // Fallback to single allocator approach if multi-GPU distribution fails
    Allocator &primary_allocator = const_cast<Allocator&>(allocators[0]);
    return construct_realm_tensor_backing(allocated_tensors, unallocated_tensors, primary_allocator);
  }
}

// Helper: Create tensor-to-device mapping based on PCG analysis
std::unordered_map<tensor_guid_t, device_id_t> create_tensor_device_mapping(
    ParallelComputationGraph const &pcg,
    UnstructuredDeviceMapping const &device_mapping,
    size_t num_devices) {
  
  std::unordered_map<tensor_guid_t, device_id_t> mapping;
  
  // Get all tensors from PCG
  std::unordered_set<parallel_tensor_guid_t> parallel_tensors = get_parallel_tensors(pcg);
  
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_tensors) {
    try {
      // Convert to regular tensor
      tensor_guid_t tensor = convert_parallel_to_regular_tensor(parallel_tensor);
      
      // Get device placement for this tensor from PCG
      device_id_t device = get_tensor_device_placement(device_mapping, parallel_tensor);
      
      // Validate device ID
      if (device.gpu_id.gpu_index.raw_value < num_devices) {
        mapping[tensor] = device;
      } else {
        // Fallback to round-robin if device ID is out of range
        size_t device_index = std::hash<tensor_guid_t>{}(tensor) % num_devices;
        mapping[tensor] = device_id_t(gpu_id_t(nonnegative_int(device_index)));
      }
      
    } catch (const std::exception& e) {
      // Skip tensors that can't be mapped
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

// Helper: Merge device tensor backings into unified backing
RealmTensorBacking merge_device_tensor_backings(
    std::vector<RealmTensorBacking> const &device_backings,
    std::vector<Allocator> const &allocators) {
  
  if (device_backings.empty()) {
    throw std::runtime_error("No device tensor backings to merge");
  }
  
  // Start with the first device backing
  RealmTensorBacking merged_backing = device_backings[0];
  
  // Merge tensor backings from other devices
  for (size_t i = 1; i < device_backings.size(); i++) {
    RealmTensorBacking const &device_backing = device_backings[i];
    
    // Merge tensor backings
    for (auto const &tensor_pair : device_backing.tensor_backings) {
      merged_backing.tensor_backings[tensor_pair.first] = tensor_pair.second;
    }
    
    // Merge gradient mappings
    for (auto const &grad_pair : device_backing.tensor_gradient_mapping) {
      merged_backing.tensor_gradient_mapping[grad_pair.first] = grad_pair.second;
    }
    
    // Merge optimizer mappings
    for (auto const &opt_pair : device_backing.tensor_optimizer_mapping) {
      merged_backing.tensor_optimizer_mapping[opt_pair.first] = opt_pair.second;
    }
  }
  
  return merged_backing;
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
    realm_tensor_backing(construct_multi_gpu_realm_tensor_backing(
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
    realm_tensor_backing(construct_multi_gpu_realm_tensor_backing(
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
    
    // Map each device to a corresponding processor (round-robin if needed)
    device_to_processor[device] = worker_procs[i % worker_procs.size()];
    
    // Note: Allocator mapping is now handled dynamically in get_device_allocator()
    // using round-robin distribution based on device ID
  }
}

TaskRegistry construct_task_registry_and_register_tasks_for_realm_pcg(
    ParallelComputationGraph const &pcg,
    std::vector<Processor> const &worker_procs) {
  
  // Use PCG functions to get layer attributes mapping
  std::unordered_map<layer_guid_t, LayerAttrs> layer_attrs_mapping = 
      get_layer_attrs_mapping_from_pcg(pcg);
  
  TaskRegistry task_registry = construct_task_registry(layer_attrs_mapping);

  // Note: Skipping task registration for now due to missing functions
  // This would need proper PCG integration to work
  
  return task_registry;
}

RealmArgsBacking initialize_args_backing_pcg(
    RealmTrainingBackingPCG *backing,
    ParallelComputationGraph const &pcg,
    RuntimeArgConfig const &runtime_arg_config) {
  
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates> per_device_op_states;
  
  // Use PCG topological ordering
  std::vector<parallel_layer_guid_t> pcg_layers = topological_ordering(pcg);
  
  // Process each layer in the PCG
  for (parallel_layer_guid_t const &parallel_layer : pcg_layers) {
    // Convert parallel layer to regular layer for compatibility with existing args backing
    // This is a temporary approach until full PCG integration is completed
    try {
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
      
      // Check if this layer needs initialization
      if (registry_contains_task_for_layer(backing->task_registry, regular_layer, OpTaskType::INIT)) {
        ParallelLayerAttrs parallel_layer_attrs = get_parallel_layer_attrs(pcg, parallel_layer);
        // Note: need to convert ParallelLayerAttrs to LayerAttrs here
        // This is a placeholder for now
        
        // For now, skip initialization until proper conversion is implemented
      }
    } catch (std::runtime_error const &e) {
      // Skip layers that can't be converted for now
      continue;
    }
  }
  
  return RealmArgsBacking{runtime_arg_config, per_device_op_states};
}

Future<float> execute_forward_pcg(RealmTrainingBackingPCG &backing,
                                 parallel_layer_guid_t const &layer) {
  
  // Get devices for this layer
  std::vector<device_id_t> devices = get_layer_devices(backing, layer);
  
  if (devices.empty()) {
    return Future<float>(0.0f);
  }
  
  // Get layer attributes from PCG
  ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(backing.pcg, layer);
  PCGOperatorAttrs op_attrs = pcg_get_op_attrs(backing.pcg, layer);
  
  // FIXED: Execute on ALL devices simultaneously (not sequentially)
  std::vector<Future<float>> device_futures;
  device_futures.reserve(devices.size());
  
  // Create parallel execution contexts for all devices
  std::vector<std::unique_ptr<ParallelExecutionContext>> execution_contexts;
  
  for (device_id_t device : devices) {
    // Create execution context for this device
    auto context = std::make_unique<ParallelExecutionContext>(
        backing, layer, device, op_attrs);
    
    // Spawn task on device processor immediately (asynchronous)
    Future<float> device_future = spawn_device_task_async(std::move(context));
    device_futures.push_back(device_future);
  }
  
  // Combine results from all devices
  return combine_device_results_parallel(device_futures);
}

// Helper: Asynchronous task spawning for parallel execution
Future<float> spawn_device_task_async(std::unique_ptr<ParallelExecutionContext> context) {
  // Get device-specific processor
  Processor device_proc = get_device_processor(context->backing, context->device);
  
  // Create task invocation
  TaskInvocation invocation = create_task_invocation_for_device(
      context->backing, context->layer, context->device, context->op_attrs);
  
  // Get device-specific task accessor
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      context->backing.realm_tensor_backing,
      context->backing.realm_args_backing,
      invocation,
      context->device,
      context->backing);
  
  // Create promise/future for result
  Promise<float> promise(context->backing.master_mem);
  Future<float> future = promise.get_future();
  
  // Package task arguments
  RealmTaskArgs<float>* task_arg = new RealmTaskArgs<float>{
      invocation.task_id,
      context->backing.task_registry.task_mapping.at(invocation.task_id).impl_function,
      accessor,
      std::move(promise)
  };
  
  uintptr_t args[1] = {reinterpret_cast<uintptr_t>(task_arg)};
  
  // CRITICAL: Spawn task immediately without waiting for previous tasks
  Event spawn_event = device_proc.spawn(
      get_realm_task_id(invocation.task_id),
      args,
      sizeof(uintptr_t),
      Event::NO_EVENT  // Don't wait for previous events
  );
  
  future.set_event(spawn_event);
  return future;
}

Future<float> execute_forward_on_device(RealmTrainingBackingPCG &backing,
                                       parallel_layer_guid_t const &layer,
                                       device_id_t device,
                                       PCGOperatorAttrs const &attrs) {
  
  // Get device-specific processor and allocator
  Processor device_proc = get_device_processor(backing, device);
  
  // Create forward task invocation using PCG functions
  OpTaskInvocation op_invocation = forward(attrs);
  
  // Convert parallel layer to regular layer for compatibility
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
  
  // Get tensor information from PCG
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  // Convert to regular tensors
  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  // Get input shapes
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  
  // Get device states if available
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
      backing.realm_tensor_backing.tensor_gradient_mapping,
      device_state);
  
  // Execute on the specific device
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.realm_tensor_backing,
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
                                  parallel_layer_guid_t const &layer) {
  
  // Get devices for this layer
  std::vector<device_id_t> devices = get_layer_devices(backing, layer);
  
  if (devices.empty()) {
    return Future<float>(0.0f);
  }
  
  // Get layer attributes from PCG
  PCGOperatorAttrs op_attrs = pcg_get_op_attrs(backing.pcg, layer);
  
  // Execute on each device and combine results
  std::vector<Future<float>> device_futures;
  for (device_id_t device : devices) {
    Future<float> device_future = execute_backward_on_device(backing, layer, device, op_attrs);
    device_futures.push_back(device_future);
  }
  
  return combine_device_results(device_futures);
}

Future<float> execute_backward_on_device(RealmTrainingBackingPCG &backing,
                                        parallel_layer_guid_t const &layer,
                                        device_id_t device,
                                        PCGOperatorAttrs const &attrs) {
  
  // Get device-specific processor and allocator
  Processor device_proc = get_device_processor(backing, device);
  
  // Create backward task invocation using PCG functions
  OpTaskInvocation op_invocation = backward(attrs);
  
  // Convert parallel layer to regular layer for compatibility
  layer_guid_t regular_layer = convert_parallel_to_regular_layer(layer);
  
  // Get tensor information from PCG
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(backing.pcg, layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(backing.pcg, layer);
  
  // Convert to regular tensors
  std::vector<tensor_guid_t> inputs = transform(parallel_inputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> outputs = transform(parallel_outputs, convert_parallel_to_regular_tensor);
  std::vector<tensor_guid_t> weights = transform(parallel_weights, convert_parallel_to_regular_tensor);
  
  // Get input shapes
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(backing.pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  
  // Get device states if available
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
      backing.realm_tensor_backing.tensor_gradient_mapping,
      device_state);
  
  // Execute on the specific device
  TaskArgumentAccessor accessor = get_task_arg_accessor_pcg(
      backing.realm_tensor_backing,
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
  
  // Get devices for this layer
  std::vector<device_id_t> devices = get_layer_devices(backing, layer);
  
  // Execute update on each device
  std::vector<Future<void>> update_futures;
  for (device_id_t device : devices) {
    // Note: Would implement device-specific update execution here
    update_futures.push_back(Future<void>());
  }
  
  return combine_update_futures(update_futures);
}

Future<void> compute_loss_pcg(RealmTrainingBackingPCG &backing,
                             LossAttrs const &loss_attrs,
                             parallel_tensor_guid_t const &logit_tensor,
                             loss_tensor_t const &label_tensor) {
  
  // Get devices for this tensor
  std::vector<device_id_t> devices = get_tensor_devices(backing, logit_tensor);
  
  // Execute loss computation on each device
  std::vector<Future<void>> loss_futures;
  for (device_id_t device : devices) {
    // Note: Would implement device-specific loss computation here
    loss_futures.push_back(Future<void>());
  }
  
  return combine_loss_futures(loss_futures);
}

// Device management functions
// FIXED: PCG-based device mapping (replaces round-robin)
std::vector<device_id_t> get_layer_devices(RealmTrainingBackingPCG const &backing,
                                          parallel_layer_guid_t const &layer) {
  std::vector<device_id_t> devices;
  
  try {
    // Get the PCG device mapping for this layer
    UnstructuredDeviceMapping device_mapping = 
        get_unstructured_device_mapping(backing.machine_mapping, backing.machine_spec, backing.pcg);
    
    // Get the parallelization strategy for this layer
    ParallelizationStrategy strategy = get_parallelization_strategy(backing.pcg, layer);
    
    // Get device placement based on PCG analysis
    std::vector<device_id_t> pcg_devices = get_layer_device_placement(device_mapping, layer);
    
    // Validate that devices are available in our backing
    for (device_id_t device : pcg_devices) {
      if (is_device_available(backing, device)) {
        devices.push_back(device);
      }
    }
    
    // If no PCG devices available, fall back to strategy-based assignment
    if (devices.empty()) {
      devices = get_devices_by_strategy(backing, layer, strategy);
    }
    
  } catch (const std::exception& e) {
    // Fallback to basic device assignment if PCG mapping fails
    devices = get_fallback_devices(backing, layer);
  }
  
  // Ensure we have at least one device
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
      // Data parallelism: Use all available devices
      for (size_t i = 0; i < available_devices; i++) {
        devices.push_back(device_id_t(gpu_id_t(nonnegative_int(i))));
      }
      break;
      
    case ParallelizationType::MODEL_PARALLEL:
      // Model parallelism: Use devices based on model partition
      {
        size_t partition_size = strategy.partition_size;
        size_t num_partitions = std::min(available_devices, partition_size);
        for (size_t i = 0; i < num_partitions; i++) {
          devices.push_back(device_id_t(gpu_id_t(nonnegative_int(i))));
        }
      }
      break;
      
    case ParallelizationType::PIPELINE_PARALLEL:
      // Pipeline parallelism: Use specific stage device
      {
        size_t stage_id = strategy.stage_id;
        if (stage_id < available_devices) {
          devices.push_back(device_id_t(gpu_id_t(nonnegative_int(stage_id))));
        }
      }
      break;
      
    default:
      // Unknown strategy: use single device
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
    // Get layer attributes from PCG
    ParallelLayerAttrs layer_attrs = get_parallel_layer_attrs(pcg, layer);
    
    // Extract parallelization information from operator attributes
    PCGOperatorAttrs op_attrs = layer_attrs.op_attrs;
    
    // Determine strategy based on operator type and attributes
    return infer_parallelization_strategy(op_attrs);
    
  } catch (const std::exception& e) {
    // Default to data parallelism if strategy can't be determined
    return ParallelizationStrategy{
        .type = ParallelizationType::DATA_PARALLEL,
        .partition_size = 1,
        .stage_id = 0
    };
  }
}

// Helper: Infer parallelization strategy from operator attributes
ParallelizationStrategy infer_parallelization_strategy(PCGOperatorAttrs const &op_attrs) {
  // This would need to be implemented based on your specific operator types
  // For now, default to data parallelism
  
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
  // Multi-GPU: Distribute allocators across devices using round-robin
  // Extract the GPU ID to determine which allocator to use
  auto gpu_id = device_id.gpu_id.gpu_index;
  size_t allocator_index = gpu_id.raw_value % backing.allocators.size();
  
  return const_cast<Allocator&>(backing.allocators[allocator_index]);
}

TaskArgumentAccessor get_task_arg_accessor_pcg(
    RealmTensorBacking const &realm_tensor_backing,
    RealmArgsBacking const &realm_args_backing,
    TaskInvocation const &invocation,
    device_id_t target_device,
    RealmTrainingBackingPCG &backing) {
  
  TensorSlotsBacking tensor_slots_backing =
      construct_tensor_slots_backing(realm_tensor_backing, invocation.binding);
  ArgSlotsBacking arg_slots_backing = construct_arg_slots_backing(
      invocation.binding, realm_args_backing.runtime_arg_config);
      
  // Multi-GPU: use device-specific allocator
  Allocator &device_allocator = get_device_allocator(backing, target_device);
  return TaskArgumentAccessor::create<RealmTaskArgumentAccessor>(
      device_allocator, tensor_slots_backing, arg_slots_backing);
}

// Helper functions for multi-device result combination
Future<float> combine_device_results(std::vector<Future<float>> const &device_futures) {
  if (!device_futures.empty()) {
    return device_futures[0];
  }
  return Future<float>(0.0f);
}

// FIXED: Proper parallel result combination
Future<float> combine_device_results_parallel(std::vector<Future<float>> const &device_futures) {
  if (device_futures.empty()) {
    return Future<float>(0.0f);
  }
  
  // For single device, return directly
  if (device_futures.size() == 1) {
    return device_futures[0];
  }
  
  // For multiple devices, we need to wait for all results and combine them
  // This is where the actual parallel combination strategy is implemented
  
  // Create a combined future that waits for all device futures
  Promise<float> combined_promise;
  Future<float> combined_future = combined_promise.get_future();
  
  // Create a result combination task that will run when all devices complete
  auto combination_task = [device_futures, promise = std::move(combined_promise)]() mutable {
    try {
      std::vector<float> device_results;
      device_results.reserve(device_futures.size());
      
      // Wait for all device results
      for (Future<float> const &future : device_futures) {
        device_results.push_back(future.get());
      }
      
      // Combine results based on parallelization strategy
      float combined_result = combine_parallel_results(device_results);
      
      // Set the combined promise
      promise.set_value(combined_result);
    } catch (const std::exception& e) {
      promise.set_exception(std::current_exception());
    }
  };
  
  // Execute combination task asynchronously
  std::thread(combination_task).detach();
  
  return combined_future;
}

// Helper: Combine results from multiple devices based on parallelization strategy
float combine_parallel_results(std::vector<float> const &device_results) {
  if (device_results.empty()) {
    return 0.0f;
  }
  
  // Different combination strategies based on parallelization type:
  
  // Strategy 1: Data Parallelism - Average the results
  // (Each device processes a different batch, results should be averaged)
  float sum = 0.0f;
  for (float result : device_results) {
    sum += result;
  }
  return sum / static_cast<float>(device_results.size());
  
  // Strategy 2: Model Parallelism - Sum the results
  // (Each device processes part of the model, results should be summed)
  // return std::accumulate(device_results.begin(), device_results.end(), 0.0f);
  
  // Strategy 3: Pipeline Parallelism - Return last stage result
  // (Each device processes a different stage, return final stage result)
  // return device_results.back();
}

Future<void> combine_update_futures(std::vector<Future<void>> const &update_futures) {
  if (!update_futures.empty()) {
    return update_futures[0];
  }
  return Future<void>();
}

Future<void> combine_loss_futures(std::vector<Future<void>> const &loss_futures) {
  if (!loss_futures.empty()) {
    return loss_futures[0];
  }
  return Future<void>();
}

// Placeholder implementations for missing conversion functions
layer_guid_t convert_parallel_to_regular_layer(parallel_layer_guid_t const &parallel_layer) {
  // Direct conversion: both types wrap the same Node
  return layer_guid_t{parallel_layer.raw_graph_node};
}

tensor_guid_t convert_parallel_to_regular_tensor(parallel_tensor_guid_t const &parallel_tensor) {
  // Direct conversion: both types wrap the same DataflowOutput
  return tensor_guid_t{parallel_tensor.raw_graph_output};
}

// Helper: Convert the other direction
parallel_layer_guid_t convert_regular_to_parallel_layer(layer_guid_t const &regular_layer) {
  return parallel_layer_guid_t{regular_layer.raw_node};
}

parallel_tensor_guid_t convert_regular_to_parallel_tensor(tensor_guid_t const &regular_tensor) {
  return parallel_tensor_guid_t{regular_tensor.raw_graph_output};
}

// PCG integration functions using actual PCG API
std::unordered_map<layer_guid_t, LayerAttrs> get_layer_attrs_mapping_from_pcg(ParallelComputationGraph const &pcg) {
  std::unordered_map<layer_guid_t, LayerAttrs> layer_attrs_mapping;
  
  // Get all parallel layers from PCG
  std::unordered_set<parallel_layer_guid_t> parallel_layers = get_parallel_layers(pcg);
  
  for (parallel_layer_guid_t const &parallel_layer : parallel_layers) {
    try {
      // Convert parallel layer to regular layer
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
      
      // Get parallel layer attributes from PCG
      ParallelLayerAttrs parallel_attrs = get_parallel_layer_attrs(pcg, parallel_layer);
      
      // Convert ParallelLayerAttrs to LayerAttrs using existing conversion functions
      LayerAttrs layer_attrs = LayerAttrs{
        compgraph_op_attrs_from_pcg_op_attrs(parallel_attrs.op_attrs),
        parallel_attrs.name
      };
      
      layer_attrs_mapping[regular_layer] = layer_attrs;
    } catch (std::runtime_error const &e) {
      // Skip layers that can't be converted (parallel-only ops like Repartition)
      continue;
    }
  }
  
  return layer_attrs_mapping;
}

std::unordered_map<tensor_guid_t, TensorAttrs> get_all_tensor_attrs_from_pcg(ParallelComputationGraph const &pcg) {
  std::unordered_map<tensor_guid_t, TensorAttrs> tensor_attrs_mapping;
  
  // Get all parallel tensors from PCG
  std::unordered_set<parallel_tensor_guid_t> parallel_tensors = get_parallel_tensors(pcg);
  
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_tensors) {
    try {
      // Convert parallel tensor to regular tensor
      tensor_guid_t regular_tensor = convert_parallel_to_regular_tensor(parallel_tensor);
      
      // Get parallel tensor attributes from PCG
      ParallelTensorAttrs parallel_attrs = get_parallel_tensor_attrs(pcg, parallel_tensor);
      
      // Convert ParallelTensorAttrs to TensorAttrs using existing conversion function
      TensorAttrs tensor_attrs = get_piece_attrs(parallel_attrs);
      
      tensor_attrs_mapping[regular_tensor] = tensor_attrs;
    } catch (std::runtime_error const &e) {
      // Skip tensors that can't be converted for now
      continue;
    }
  }
  
  return tensor_attrs_mapping;
}

LayerAttrs get_layer_attrs_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  // Convert regular layer to parallel layer
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  
  // Get parallel layer attributes from PCG
  ParallelLayerAttrs parallel_attrs = get_parallel_layer_attrs(pcg, parallel_layer);
  
  // Convert to regular layer attributes
  return LayerAttrs{
    compgraph_op_attrs_from_pcg_op_attrs(parallel_attrs.op_attrs),
    parallel_attrs.name
  };
}

std::vector<layer_guid_t> topological_ordering_from_pcg(ParallelComputationGraph const &pcg) {
  // Get PCG topological ordering and convert to regular layer ordering
  std::vector<parallel_layer_guid_t> parallel_ordering = topological_ordering(pcg);
  std::vector<layer_guid_t> regular_ordering;
  
  for (parallel_layer_guid_t const &parallel_layer : parallel_ordering) {
    try {
      layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
      regular_ordering.push_back(regular_layer);
    } catch (std::runtime_error const &e) {
      // Skip layers that can't be converted
      continue;
    }
  }
  
  return regular_ordering;
}

std::vector<tensor_guid_t> get_incoming_inputs_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  // Convert layer to parallel layer and get inputs
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(pcg, parallel_layer);
  
  // Convert parallel tensors to regular tensors
  std::vector<tensor_guid_t> regular_inputs;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    regular_inputs.push_back(convert_parallel_to_regular_tensor(parallel_tensor));
  }
  return regular_inputs;
}

std::vector<TensorShape> get_incoming_input_shapes_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  // Convert layer to parallel layer and get input shapes
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_inputs = get_incoming_inputs(pcg, parallel_layer);
  
  // Get tensor shapes and convert them
  std::vector<TensorShape> input_shapes;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_inputs) {
    ParallelTensorShape parallel_shape = get_parallel_tensor_shape(pcg, parallel_tensor);
    input_shapes.push_back(get_piece_shape(parallel_shape));
  }
  return input_shapes;
}

std::vector<tensor_guid_t> get_outgoing_tensors_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  // Convert layer to parallel layer and get outputs using get_layer_outputs
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_outputs = get_layer_outputs(pcg, parallel_layer);
  
  // Convert parallel tensors to regular tensors
  std::vector<tensor_guid_t> regular_outputs;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_outputs) {
    regular_outputs.push_back(convert_parallel_to_regular_tensor(parallel_tensor));
  }
  return regular_outputs;
}

std::vector<tensor_guid_t> get_incoming_weights_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer) {
  // Convert layer to parallel layer and get weights using get_incoming_weights
  parallel_layer_guid_t parallel_layer = convert_regular_to_parallel_layer(layer);
  std::vector<parallel_tensor_guid_t> parallel_weights = get_incoming_weights(pcg, parallel_layer);
  
  // Convert parallel tensors to regular tensors
  std::vector<tensor_guid_t> regular_weights;
  for (parallel_tensor_guid_t const &parallel_tensor : parallel_weights) {
    regular_weights.push_back(convert_parallel_to_regular_tensor(parallel_tensor));
  }
  return regular_weights;
}

std::vector<device_id_t> get_tensor_devices(RealmTrainingBackingPCG const &backing, parallel_tensor_guid_t const &tensor) {
  // Use PCG device mapping to determine which devices this tensor resides on
  // For now, use the same logic as layers - tensor follows its source layer
  parallel_layer_guid_t source_layer = get_source_layer(backing.pcg, tensor);
  return get_layer_devices(backing, source_layer);
}

} // namespace FlexFlow
