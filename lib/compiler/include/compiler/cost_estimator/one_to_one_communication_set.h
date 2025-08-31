#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_ONE_TO_ONE_COMMUNICATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_ONE_TO_ONE_COMMUNICATION_SET_H

#include "op-attrs/operator_task_space_dim_idx_t.dtg.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/operator_space_to_machine_space_mapping.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "utils/orthotope/dim_domain_mapping.h"
#include "compiler/cost_estimator/one_to_one_communication_set.dtg.h"

namespace FlexFlow {

OneToOneCommunicationSet 
  one_to_one_communication_set_from_composition(
    OperatorSpaceToMachineSpaceMapping const &pre_map,
    OperatorTaskSpaceToOperatorTaskSpaceMapping const &operator_task_space_map,
    OperatorSpaceToMachineSpaceMapping const &post_map);

OneToOneCommunicationSet
  pcg_get_communication_for_edge(
    ParallelComputationGraphEdge const &edge,
    ParallelComputationGraph const &pcg,
    MachineComputeSpecification const &machine_compute_specification,
    MachineView const &src_mv,
    MachineView const &dst_mv);

} // namespace FlexFlow

#endif
