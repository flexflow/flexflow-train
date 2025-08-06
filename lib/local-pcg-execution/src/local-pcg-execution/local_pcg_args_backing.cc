#include "local-pcg-execution/local_pcg_args_backing.h"

namespace FlexFlow {

TaskArgumentAccessor
    get_task_arg_accessor(LocalParallelTensorBacking const &local_tensor_backing,
                          RuntimeArgConfig const &runtime_arg_config,
                          TaskInvocation const &invocation,
                          Allocator &allocator) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
