#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TENSOR_BACKING_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "local-execution/atomic_task_binding.dtg.h"
#include "local-execution/atomic_task_invocation/atomic_task_invocation.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "local-execution/tensor_slot_backing.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_config.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_task_invocation.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "task-spec/symbolic/symbolic_training_tensor_guid_t.dtg.h"
#include "local-execution/atomic_training_tensor_guid_t.dtg.h"

namespace FlexFlow {

LocalTensorBacking local_tensor_backing_for_tensor(
  symbolic_training_tensor_guid_t);

LocalTensorBacking
  merge_local_tensor_backings(LocalTensorBacking const &,
                              LocalTensorBacking const &);

AtomicTaskBinding 
  lower_local_runtime_task_binding_to_atomic_task_binding(
    LocalTensorBacking const &,
    RuntimeTaskBinding const &,
    RuntimeArgConfig const &);

AtomicTaskInvocation 
  lower_local_runtime_task_invocation_to_atomic_task_invocation(
    LocalTensorBacking const &,
    RuntimeTaskInvocation const &,
    RuntimeArgConfig const &);

} // namespace FlexFlow

#endif
