#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_EXECUTE_TASK_FOR_LAYER_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_EXECUTE_TASK_FOR_LAYER_H

#include "local-execution/local_ready_to_launch_task.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "task-spec/symbolic/symbolic_cg_op_attrs_and_training_signature_with_shapes.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph_from_cg_conversion.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

LocalReadyToLaunchTask prepare_runtime_task_invocation(
  RuntimeTaskInvocation const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  RuntimeArgConfig const &);

std::optional<DeviceSpecificPerDeviceOpState> execute_init_for_layer(
  symbolic_layer_guid_t,
  SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

std::optional<milliseconds_t> execute_forward_for_layer(
  symbolic_layer_guid_t,
  SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

std::optional<milliseconds_t> execute_backward_for_layer(
  symbolic_layer_guid_t,
  SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

void execute_compute_loss(
  TrainingSymbolicComputationGraph const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

void execute_update_for_layer(
  symbolic_layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  OptimizerAttrs const &,
  Allocator &,
  RuntimeArgConfig const &);

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
  execute_forward_pass(
    TrainingSymbolicComputationGraphFromCgConversion const &training_cg,
    LocalTensorBacking const &local_tensor_backing,
    LocalAtomicTensorBacking const &local_atomic_tensor_backing,
    Allocator &,
    LocalTaskRegistry const &,
    RuntimeArgConfig const &);

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
  execute_backward_pass(
    TrainingSymbolicComputationGraphFromCgConversion const &training_cg,
    LocalTensorBacking const &local_tensor_backing,
    LocalAtomicTensorBacking const &local_atomic_tensor_backing,
    Allocator &,
    LocalTaskRegistry const &,
    RuntimeArgConfig const &);


} // namespace FlexFlow

#endif
