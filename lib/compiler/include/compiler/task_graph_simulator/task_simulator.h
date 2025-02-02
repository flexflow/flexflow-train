#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TASK_SIMULATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TASK_SIMULATOR_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/machine_mapping/machine_mapping.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {
float task_simulator_estimate_forward_pass_time(
    ParallelComputationGraph const &pcg,
    CostEstimator const &estimator,
    MachineMapping const &machine_mapping,
    MachineSpecification const &machine_spec);

} // namespace FlexFlow

#endif
