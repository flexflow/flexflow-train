#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "local-execution/atomic_task_invocation.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "task-spec/task_argument_accessor.h"
#include "task-spec/symbolic_training_tensor_guid_t.dtg.h"
#include "local-execution/atomic_training_tensor_guid_t.dtg.h"

namespace FlexFlow {

AtomicTaskInvocation 
  lower_local_runtime_task_invocation_to_atomic_task_invocation(
    LocalTensorBacking const &,
    RuntimeTaskInvocation const &,
    RuntimeArgConfig const &);

LocalTensorBacking construct_local_tensor_backing(
    std::unordered_map<symbolic_training_tensor_guid_t, TensorShape> const
        &training_tensor_shapes,
    // std::unordered_map<symbolic_training_tensor_guid_t, GenericTensorAccessorW> const
    //     &preallocated_tensors,
    Allocator &);

} // namespace FlexFlow

#endif
