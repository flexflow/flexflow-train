#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/task_argument_accessor.h"
#include "task-spec/task_binding.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/symbolic_training_tensor_guid_t.dtg.h"
#include "local-execution/atomic_training_tensor_guid_t.dtg.h"

namespace FlexFlow {

LocalTensorBacking construct_local_tensor_backing(
    std::unordered_map<symbolic_training_tensor_guid_t, TensorShape> const
        &training_tensor_shapes,
    std::unordered_map<symbolic_training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated_tensors,
    Allocator &);

TaskArgumentAccessor get_task_arg_accessor_for_invocation(LocalTensorBacking const &,
                                           RuntimeArgConfig const &,
                                           TaskInvocation const &,
                                           Allocator &);

atomic_training_tensor_guid_t
    get_atomic_tensor_for_symbolic_tensor(LocalTensorBacking const &,
                                          symbolic_training_tensor_guid_t);

std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
    construct_tensor_slots_backing_for_binding(LocalTensorBacking const &,
                                               TaskBinding const &);

} // namespace FlexFlow

#endif
