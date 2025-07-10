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

namespace FlexFlow {

class GradientTensorSource;
class OptimizerTensorSource;

using PerLayerElapsedTimePCG =
    std::unordered_map<parallel_layer_guid_t, std::optional<float>>;

struct RealmTrainingBackingPCG {
  RealmTrainingBackingPCG(Realm::Processor master_proc,
                         std::vector<Realm::Processor> const &worker_procs,
                         std::vector<Allocator> const &allocators,
                         AllocatedTensors const &allocated_tensors,
                         GradientTensorSource &gradient_tensor_source,
                         ParallelComputationGraph const &pcg,
                         MachineMapping const &machine_mapping,
                         MachineSpecification const &machine_spec,
                         RuntimeArgConfig const &runtime_arg_config);

  RealmTrainingBackingPCG(Realm::Processor master_proc,
                         std::vector<Realm::Processor> const &worker_procs,
                         std::vector<Allocator> const &allocators,
                         AllocatedTensors const &allocated_tensors,
                         GradientTensorSource &gradient_tensor_source,
                         OptimizerTensorSource &optimizer_tensor_source,
                         ParallelComputationGraph const &pcg,
                         MachineMapping const &machine_mapping,
                         MachineSpecification const &machine_spec,
                         RuntimeArgConfig const &runtime_arg_config,
                         OptimizerAttrs const &optimizer_attrs);

  // Initialize device mappings based on PCG information
  void initialize_device_mappings();

public:
  // runtime - enhanced for multi-device support
  Realm::Processor master_proc;
  Realm::Event master_event;
  Realm::Memory master_mem;
  std::vector<Realm::Processor> worker_procs;
  std::vector<Realm::Event> worker_events;
  std::vector<Allocator> allocators;

  // PCG-specific components
  ParallelComputationGraph pcg;
  MachineMapping machine_mapping;
  MachineSpecification machine_spec;
  TaskRegistry task_registry;

  // Enhanced backing with device-aware mapping
  RealmTensorBacking realm_tensor_backing;
  RealmArgsBacking realm_args_backing;

  // Device mapping functionality
  std::unordered_map<parallel_layer_guid_t, std::vector<device_id_t>> layer_to_devices;
  std::unordered_map<device_id_t, Realm::Processor> device_to_processor;
};

// Multi-GPU aware task registry construction
TaskRegistry construct_task_registry_and_register_tasks_for_realm_pcg(
    ParallelComputationGraph const &pcg,
    std::vector<Realm::Processor> const &worker_procs);

// Multi-GPU tensor backing construction - distributes tensors across allocators
RealmTensorBacking construct_multi_gpu_realm_tensor_backing(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    std::vector<Allocator> const &allocators,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec,
    ParallelComputationGraph const &pcg);

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
                                       ComputationGraphOpAttrs const &attrs);

Future<float> execute_backward_pcg(RealmTrainingBackingPCG &backing,
                                  parallel_layer_guid_t const &layer);

// Device-specific backward execution
Future<float> execute_backward_on_device(RealmTrainingBackingPCG &backing,
                                        parallel_layer_guid_t const &layer,
                                        device_id_t device,
                                        ComputationGraphOpAttrs const &attrs);

Future<void> compute_loss_pcg(RealmTrainingBackingPCG &backing, 
                             LossAttrs const &loss_attrs,
                             parallel_tensor_guid_t const &logit_tensor,
                             loss_tensor_t const &label_tensor);

Future<void> execute_update_pcg(RealmTrainingBackingPCG &backing,
                               parallel_layer_guid_t const &layer,
                               OptimizerAttrs const &optimizer_attrs);

// Device management functions
std::vector<device_id_t> get_layer_devices(RealmTrainingBackingPCG const &backing,
                                          parallel_layer_guid_t const &layer);

Realm::Processor get_device_processor(RealmTrainingBackingPCG const &backing,
                                     device_id_t device_id);

Allocator &get_device_allocator(RealmTrainingBackingPCG &backing,
                               device_id_t device_id);

// Multi-GPU task argument accessor
TaskArgumentAccessor get_task_arg_accessor_pcg(RealmTensorBacking const &realm_tensor_backing,
                                              RealmArgsBacking const &realm_args_backing,
                                              TaskInvocation const &invocation,
                                              device_id_t target_device,
                                              RealmTrainingBackingPCG &backing);

// Multi-device result combination functions
Future<float> combine_device_results(std::vector<Future<float>> const &device_futures);
Future<void> combine_update_futures(std::vector<Future<void>> const &update_futures);
Future<void> combine_loss_futures(std::vector<Future<void>> const &loss_futures);

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

} // namespace FlexFlow

#endif
