#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PCG_TASK_GRAPH_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PCG_TASK_GRAPH_H

#include "compiler/task_graph_simulator/pcg_task_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

PCGTaskGraph get_pcg_task_graph(ParallelComputationGraph const &pcg);

} // namespace FlexFlow

#endif // PCG_TASK_GRAPH_H
