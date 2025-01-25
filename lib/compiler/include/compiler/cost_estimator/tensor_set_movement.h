#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TENSOR_SET_MOVEMENT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TENSOR_SET_MOVEMENT_H

#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"

namespace FlexFlow {

TensorSetMovement get_tensor_set_movement_from_pcg_edge(
    ParallelComputationGraphEdge const &edge,
    ParallelComputationGraph const &pcg,
    MachineView const &src_mv,
    MachineView const &dst_mv);

} // namespace FlexFlow

#endif
