#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PCG_TASK_GRAPH_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PCG_TASK_GRAPH_H

#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "compiler/task_graph_simulator/pcg_task_graph.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

PCGTaskGraph get_pcg_task_graph(ParallelComputationGraph const &pcg,
                                MachineMapping const &machine_mapping,
                                MachineSpecification const &machine_spec);

} // namespace FlexFlow

#endif
