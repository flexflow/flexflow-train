#include "local-execution/local_atomic_tensor_backing.h"

namespace FlexFlow {

TaskArgumentAccessor get_task_arg_accessor_for_atomic_task_invocation(
  LocalAtomicTensorBacking const &,
  RuntimeArgConfig const &,
  AtomicTaskInvocation const &,
  Allocator &) {

  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
