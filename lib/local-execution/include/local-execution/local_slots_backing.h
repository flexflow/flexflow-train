
#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_SLOTS_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_SLOTS_BACKING_H

#include "kernels/accessor.h"
#include "local-execution/layer_tensor_key.dtg.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/non_graph_tensor_guid_t.dtg.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/per_device_op_state.h"
#include "local-execution/runtime_arg_config.h"
#include "local-execution/task_invocation.dtg.h"
#include "local-execution/tensor_role.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

using LayerTensorBackingMap =
    std::unordered_map<LayerTensorKey, GenericTensorAccessorW>;

using TensorBackingMap =
    std::unordered_map<reduced_tensor_t, GenericTensorAccessorW>;

struct LocalSlotsBacking {
  LocalSlotsBacking(LayerTensorBackingMap const &allocated_forward_tensors,
                    TensorBackingMap const &allocated_non_graph_tensors,
                    RuntimeArgConfig const &);

public:
  void add_per_device_op_state(layer_guid_t const &,
                               DeviceSpecificDeviceStates const &);
  void allocate_layer_tensors(layer_guid_t const &,
                              ComputationGraph const &,
                              Allocator &);
  void allocate_tensors_by_role(TensorRole const &,
                                layer_guid_t const &,
                                ComputationGraph const &,
                                Allocator &);
  void allocate_optimizer_tensors(layer_guid_t const &weight_layer,
                                  tensor_guid_t const &,
                                  ComputationGraph const &,
                                  Allocator &,
                                  TaskSignature const &);
  TensorSlotsBacking construct_tensor_slots_backing(OpTaskBinding const &,
                                                    layer_guid_t const &) const;
  TensorSlotsBacking
      construct_tensor_slots_backing(TaskBinding const &,
                                     std::optional<layer_guid_t> const &) const;
  ArgSlotsBacking construct_arg_slots_backing(OpTaskBinding const &,
                                              layer_guid_t const &) const;
  ArgSlotsBacking construct_arg_slots_backing(TaskBinding const &) const;

  ConcreteArgSpec resolve_runtime_arg_ref_spec(RuntimeArgRefSpec const &) const;
  ConcreteArgSpec resolve_op_arg_ref_spec(OpArgRefSpec const &,
                                          layer_guid_t const &) const;

  GenericTensorAccessorW const &
      get_tensor_backing(TensorType const &,
                         reduced_tensor_t const &,
                         std::optional<layer_guid_t> const &) const;

  bool is_forward_tensor_allocated(LayerTensorKey const &) const;
  bool is_non_graph_tensor_allocated(reduced_tensor_t const &) const;

public:
  // tensors
  LayerTensorBackingMap tensor_mapping;
  LayerTensorBackingMap gradient_tensor_mapping;
  LayerTensorBackingMap optimizer_tensor_mapping;
  TensorBackingMap non_graph_tensor_mapping;
  std::unordered_map<layer_guid_t, std::vector<reduced_tensor_t>>
      input_tensor_slots;
  std::unordered_map<layer_guid_t, std::vector<reduced_tensor_t>>
      weight_tensor_slots;
  std::unordered_map<layer_guid_t, std::vector<reduced_tensor_t>>
      output_tensor_slots;
  std::unordered_map<layer_guid_t, std::vector<reduced_tensor_t>>
      weight_optimizer_tensor_guids;

  // arguments
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
      per_device_op_states;
  RuntimeArgConfig runtime_arg_config;
};

} // namespace FlexFlow

#endif
