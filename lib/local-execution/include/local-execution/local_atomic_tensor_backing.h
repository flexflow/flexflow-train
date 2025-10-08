#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_ATOMIC_TENSOR_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_ATOMIC_TENSOR_BACKING_H

#include "kernels/allocation.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "local-execution/atomic_task_invocation.dtg.h"
#include "task-spec/task_argument_accessor.h"

namespace FlexFlow {

TaskArgumentAccessor get_task_arg_accessor_for_atomic_task_invocation(
  LocalAtomicTensorBacking const &,
  RuntimeArgConfig const &,
  AtomicTaskInvocation const &,
  Allocator &);

} // namespace FlexFlow

#endif
