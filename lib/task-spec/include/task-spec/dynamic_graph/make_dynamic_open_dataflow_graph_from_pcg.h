#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_FROM_PCG_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_FROM_PCG_H

#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

DynamicOpenDataflowGraph
    make_dynamic_open_dataflow_graph_from_pcg(ParallelComputationGraph const &);

} // namespace FlexFlow

#endif
