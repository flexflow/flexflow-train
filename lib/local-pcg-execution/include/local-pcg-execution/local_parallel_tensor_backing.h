#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PARALLEL_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PARALLEL_TENSOR_BACKING_H

#include "compiler/mapped_operator_task_group.h"
#include "compiler/operator_task_signature.dtg.h"
#include "kernels/allocation.h"
#include "local-execution/atomic_task_invocation.dtg.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "local-pcg-execution/local_parallel_tensor_backing.dtg.h"
#include "local-pcg-execution/parallel_tensor_accessors_w.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "task-spec/task_binding.h"
#include "task-spec/training_tensor_slot_id_t.dtg.h"
#include "task-spec/training_parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

std::unordered_map<MachineSpaceCoordinate, AtomicTaskInvocation>
  lower_parallel_runtime_task_invocation_to_atomic_task_invocation_group(
    LocalParallelTensorBacking const &,
    RuntimeTaskInvocation const &,
    RuntimeArgConfig const &,
    MappedOperatorTaskGroup const &);

AtomicTaskInvocation 
  lower_parallel_runtime_task_invocation_to_atomic_task_invocation(
    LocalParallelTensorBacking const &,
    RuntimeTaskInvocation const &,
    RuntimeArgConfig const &,
    MachineSpaceCoordinate const &,
    OperatorTaskSignature const &);

// LocalParallelTensorBacking construct_local_parallel_tensor_backing(
//     std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorShape> const &training_ptensor_shapes,
//     std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorAccessorsW> const &preallocated_ptensors,
//     Allocator &);

} // namespace FlexFlow

#endif
