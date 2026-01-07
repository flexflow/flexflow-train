#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_ATOMIC_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_ATOMIC_TENSOR_BACKING_H

#include "kernels/allocation.h"
#include "local-execution/atomic_task_invocation.dtg.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_config.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"

namespace FlexFlow {

std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
    construct_tensor_slots_backing_for_binding(LocalAtomicTensorBacking const &,
                                               AtomicTaskBinding const &);

TaskArgumentAccessor get_task_arg_accessor_for_atomic_task_binding(
    LocalAtomicTensorBacking const &, AtomicTaskBinding const &, Allocator &);

} // namespace FlexFlow

#endif
