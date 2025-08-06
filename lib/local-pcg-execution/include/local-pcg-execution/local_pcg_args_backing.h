#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PCG_ARGS_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PCG_ARGS_BACKING_H

#include "local-execution/local_task_registry.dtg.h"
#include "local-pcg-execution/local_parallel_tensor_backing.dtg.h"
#include "local-pcg-execution/local_pcg_args_backing.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/training_parallel_computation_graph.dtg.h"

namespace FlexFlow {

TaskArgumentAccessor get_task_arg_accessor(LocalParallelTensorBacking const &,
                                           RuntimeArgConfig const &,
                                           TaskInvocation const &,
                                           Allocator &);

LocalPcgArgsBacking make_local_pcg_args_backing_for_parallel_computation_graph(
    LocalTaskRegistry const &,
    TrainingParallelComputationGraph const &,
    RuntimeArgConfig const &,
    LocalParallelTensorBacking const &,
    Allocator &);

} // namespace FlexFlow

#endif
