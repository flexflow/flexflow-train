#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_INITIALIZED_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_INITIALIZED_COMPUTATION_GRAPH_INSTANCE_H

#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "local-execution/local_device_states_backing.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_config.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph_from_cg_conversion.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct InitializedComputationGraphInstance {
public:
  LocalTensorBacking const &get_tensor_backing() const;
  LocalTaskRegistry const &get_task_registry() const;
  TrainingSymbolicComputationGraphFromCgConversion const &
      get_symbolic_training_graph_for_cg() const;
  LocalAtomicTensorBacking const &get_atomic_tensor_backing() const;
  Allocator &get_allocator() const;
  RuntimeArgConfig const &get_runtime_arg_config() const;

private:
  LocalDeviceStatesBacking per_device_op_states;
  Allocator &allocator;
  LocalAtomicTensorBacking atomic_tensor_backing;
  ComputationGraphInstance computation_graph_instance;
};

InitializedComputationGraphInstance
    initialize_computation_graph_instance(ComputationGraphInstance const &,
                                          Allocator &);

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &);

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &);

void perform_update_pass_for_computation_graph_instance(
    InitializedComputationGraphInstance const &);

} // namespace FlexFlow

#endif
