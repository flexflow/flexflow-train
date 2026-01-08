#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_RUNTIME_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_RUNTIME_ATOMIC_TASK_SHARD_BINDING_H

#include "compiler/operator_atomic_task_shard_binding.dtg.h"
#include "local-pcg-execution/runtime_atomic_task_shard_binding.dtg.h"
#include "task-spec/fwb_op_task_type.dtg.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature.dtg.h"

namespace FlexFlow {

RuntimeAtomicTaskShardBinding
    lower_op_shard_binding_to_fwd_pass_runtime_shard_binding(
        OperatorAtomicTaskShardBinding const &,
        SymbolicLayerTrainingTensorGroupSignature const &);

RuntimeAtomicTaskShardBinding
    lower_op_shard_binding_to_bwd_pass_runtime_shard_binding(
        OperatorAtomicTaskShardBinding const &,
        SymbolicLayerTrainingTensorGroupSignature const &);

RuntimeAtomicTaskShardBinding lower_op_shard_binding_to_runtime_shard_binding(
    OperatorAtomicTaskShardBinding const &,
    SymbolicLayerTrainingTensorGroupSignature const &,
    FwbOpTaskType);

} // namespace FlexFlow

#endif
