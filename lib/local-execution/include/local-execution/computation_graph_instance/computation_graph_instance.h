#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H

#include "kernels/accessor.h"
#include "local-execution/computation_graph_instance/computation_graph_instance.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include <unordered_map>

namespace FlexFlow {

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &, LocalTaskRegistry, OptimizerAttrs const &);

} // namespace FlexFlow

#endif
