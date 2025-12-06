#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_PCG_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_PCG_H

#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include <algorithm>
#include "realm.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "local-execution/allocated_tensors.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_args_backing.h"
#include "realm-backend/realm_tensor_backing.h"
#include "realm-backend/task_wrapper.h"
#include "task-spec/task_invocation.h"
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
#include "op-attrs/tensor_shape.h"
#include <unordered_set>

namespace FlexFlow {

class GradientTensorSource;
class OptimizerTensorSource;

using PerLayerElapsedTimePCG =
    std::unordered_map<parallel_layer_guid_t, std::optional<float>>;

class RealmTrainingBackingPCG {
public:
  RealmTrainingBackingPCG(
      Processor master_proc,
      std::vector<Processor> const &worker_procs,
      std::vector<Allocator> const &allocators,
      AllocatedTensors const &allocated_tensors,
      GradientTensorSource &gradient_tensor_source,
      ParallelComputationGraph const &pcg,
      MachineMapping const &machine_mapping,
      MachineSpecification const &machine_spec,
      RuntimeArgConfig const &runtime_arg_config);

  RealmTrainingBackingPCG(
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
      OptimizerAttrs const &optimizer_attrs);

  // Master processor and memory
  Processor master_proc;
  Event master_event;
  Memory master_mem;

  // Worker processors and events
  std::vector<Processor> worker_procs;
  std::vector<Event> worker_events;

  // Allocators for multi-GPU support
  std::vector<Allocator> allocators;

  // PCG-specific components
  ParallelComputationGraph pcg;
  MachineMapping machine_mapping;
  MachineSpecification machine_spec;
  TaskRegistry task_registry;

  // Device-specific tensor backings for data parallel
  std::unordered_map<device_id_t, RealmTensorBacking> device_tensor_backings;
  RealmArgsBacking realm_args_backing;

  // Device mapping functionality
  std::unordered_map<parallel_layer_guid_t, std::vector<device_id_t>> layer_to_devices;
  std::unordered_map<device_id_t, Realm::Processor> device_to_processor;

  // Helper methods for device-specific tensor access
  RealmTensorBacking const &get_device_tensor_backing(device_id_t device) const;
  RealmTensorBacking &get_device_tensor_backing(device_id_t device);
};

// Multi-GPU aware task registry construction
TaskRegistry construct_task_registry_and_register_tasks_for_realm_pcg(
    ParallelComputationGraph const &pcg,
    std::vector<Realm::Processor> const &worker_procs);

// Multi-GPU tensor backing construction - creates device-specific backings
std::unordered_map<device_id_t, RealmTensorBacking> construct_device_specific_tensor_backings(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    std::vector<Allocator> const &allocators,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec,
    ParallelComputationGraph const &pcg);

// Physical tensor replication functions
AllocatedTensors replicate_tensors_for_device(
    AllocatedTensors const &source_tensors,
    device_id_t device,
    Allocator &device_allocator);

UnallocatedTensors replicate_unallocated_tensors_for_device(
    UnallocatedTensors const &source_tensors,
    device_id_t device,
    Allocator &device_allocator);

GenericTensorAccessorW allocate_tensor_on_device(
    TensorShape const &shape,
    DataType data_type,
    Allocator &device_allocator);

size_t calculate_tensor_size(TensorShape const &shape, DataType data_type);
GenericTensorAccessorW create_tensor_accessor(
    void* device_memory,
    TensorShape const &shape,
    DataType data_type);

// Tensor data copying functions
void copy_tensor_values(GenericTensorAccessorW const &source_accessor,
                       GenericTensorAccessorW &dest_accessor);
size_t get_element_size(DataType data_type);

// Multi-GPU aware args backing initialization
RealmArgsBacking initialize_args_backing_pcg(RealmTrainingBackingPCG *backing,
                                            ParallelComputationGraph const &pcg,
                                            RuntimeArgConfig const &runtime_arg_config);

// Enhanced execution functions with device-aware scheduling
Future<float> execute_forward_pcg(RealmTrainingBackingPCG &backing,
                                 parallel_layer_guid_t const &layer);

// Device-specific forward execution
Future<float> execute_forward_on_device(RealmTrainingBackingPCG &backing,
                                       parallel_layer_guid_t const &layer,
                                       device_id_t device,
                                       PCGOperatorAttrs const &attrs);

Future<float> execute_backward_pcg(RealmTrainingBackingPCG &backing,
                                  parallel_layer_guid_t const &layer,
                                  OptimizerAttrs const &optimizer_attrs);

// Device-specific backward execution
Future<float> execute_backward_on_device(RealmTrainingBackingPCG &backing,
                                        parallel_layer_guid_t const &layer,
                                        device_id_t device,
                                        PCGOperatorAttrs const &attrs);

Future<void> compute_loss_pcg(RealmTrainingBackingPCG &backing, 
                             LossAttrs const &loss_attrs,
                             parallel_tensor_guid_t const &logit_tensor,
                             loss_tensor_t const &label_tensor);

Future<void> execute_update_pcg(RealmTrainingBackingPCG &backing,
                               parallel_layer_guid_t const &layer,
                               OptimizerAttrs const &optimizer_attrs);

// Device-specific update execution
Future<void> execute_update_on_device(RealmTrainingBackingPCG &backing,
                                     parallel_layer_guid_t const &layer,
                                     device_id_t device,
                                     OptimizerAttrs const &optimizer_attrs);

// Device-specific loss computation
Future<void> compute_loss_on_device(RealmTrainingBackingPCG &backing,
                                   LossAttrs const &loss_attrs,
                                   parallel_tensor_guid_t const &logit_tensor,
                                   loss_tensor_t const &label_tensor,
                                   device_id_t device);

// Device management functions
std::vector<device_id_t> get_layer_devices(RealmTrainingBackingPCG const &backing,
                                          parallel_layer_guid_t const &layer);

Realm::Processor get_device_processor(RealmTrainingBackingPCG const &backing,
                                     device_id_t device_id);

Allocator &get_device_allocator(RealmTrainingBackingPCG &backing,
                               device_id_t device_id);

// Multi-GPU task argument accessor
TaskArgumentAccessor get_task_arg_accessor_pcg(RealmTensorBacking const &device_tensor_backing,
                                              RealmArgsBacking const &realm_args_backing,
                                              TaskInvocation const &invocation,
                                              device_id_t target_device,
                                              RealmTrainingBackingPCG &backing);

// Multi-device result combination functions
Future<float> combine_device_results(std::vector<Future<float>> const &device_futures);
Future<float> combine_device_results_parallel(std::vector<Future<float>> const &device_futures);
Future<void> combine_update_futures(std::vector<Future<void>> const &update_futures);
Future<void> combine_loss_futures(std::vector<Future<void>> const &loss_futures);

// Parallel result combination helper
float combine_parallel_results(std::vector<float> const &device_results);

// Asynchronous task spawning for parallel execution
Future<float> spawn_device_task_async(std::unique_ptr<ParallelExecutionContext> context);

// Data parallel batch distribution functions
std::vector<TensorShape> distribute_batch_data_parallel(
    TensorShape const &original_shape,
    size_t num_devices);

std::vector<TensorShape> create_data_parallel_input_shapes(
    RealmTrainingBackingPCG const &backing,
    parallel_layer_guid_t const &layer,
    std::vector<device_id_t> const &devices);

// Data parallel gradient synchronization functions
Future<void> synchronize_gradients_data_parallel(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    std::vector<device_id_t> const &devices,
    OptimizerAttrs const &optimizer_attrs);

Future<void> synchronize_gradients_on_device(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    device_id_t device,
    OptimizerAttrs const &optimizer_attrs);

Future<void> combine_sync_futures(std::vector<Future<void>> const &sync_futures);

// All-reduce operations for gradient synchronization
Future<void> perform_all_reduce_on_device(
    RealmTrainingBackingPCG &backing,
    tensor_guid_t const &weight,
    tensor_guid_t const &gradient,
    device_id_t device,
    Processor device_proc,
    OptimizerAttrs const &optimizer_attrs);

// Weight synchronization futures combination
Future<void> combine_weight_sync_futures(std::vector<Future<void>> const &weight_sync_futures);

// Helper conversion functions
layer_guid_t convert_parallel_to_regular_layer(parallel_layer_guid_t const &parallel_layer);
tensor_guid_t convert_parallel_to_regular_tensor(parallel_tensor_guid_t const &parallel_tensor);

// PCG utility functions  
std::unordered_map<layer_guid_t, LayerAttrs> get_layer_attrs_mapping_from_pcg(ParallelComputationGraph const &pcg);
std::unordered_map<tensor_guid_t, TensorAttrs> get_all_tensor_attrs_from_pcg(ParallelComputationGraph const &pcg);
LayerAttrs get_layer_attrs_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer);
std::vector<layer_guid_t> topological_ordering_from_pcg(ParallelComputationGraph const &pcg);
std::vector<tensor_guid_t> get_incoming_inputs_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer);
std::vector<TensorShape> get_incoming_input_shapes_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer);
std::vector<tensor_guid_t> get_outgoing_tensors_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer);
std::vector<tensor_guid_t> get_incoming_weights_from_pcg(ParallelComputationGraph const &pcg, layer_guid_t const &layer);
std::vector<device_id_t> get_tensor_devices(RealmTrainingBackingPCG const &backing, parallel_tensor_guid_t const &tensor);

// Device state combination functions
DeviceSpecificDeviceStates combine_device_specific_states(
    std::vector<DeviceSpecificDeviceStates> const &device_states);

DeviceSpecificDeviceStates combine_device_states_with_tolerance(
    DeviceSpecificDeviceStates const &state1,
    DeviceSpecificDeviceStates const &state2);

PerDeviceOpState combine_layer_states_with_tolerance(
    PerDeviceOpState const &state1,
    PerDeviceOpState const &state2);

// Device state synchronization functions
Future<void> synchronize_device_states(
    RealmTrainingBackingPCG &backing,
    parallel_layer_guid_t const &layer,
    std::vector<device_id_t> const &devices);

DeviceSpecificDeviceStates get_device_state_for_layer(
    RealmTrainingBackingPCG &backing,
    layer_guid_t const &layer,
    device_id_t device);

void store_combined_device_state(
    RealmTrainingBackingPCG &backing,
    layer_guid_t const &layer,
    DeviceSpecificDeviceStates const &combined_state);

// Floating-point comparison helpers
bool float_equal_with_tolerance(float a, float b, float tolerance = 1e-6f);
bool double_equal_with_tolerance(double a, double b, double tolerance = 1e-12);
float combine_float_values_with_tolerance(float a, float b, float tolerance = 1e-6f);

} // namespace FlexFlow

#endif
