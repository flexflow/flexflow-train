#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_ONE_TO_ONE_COMMUNICATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_ONE_TO_ONE_COMMUNICATION_SET_H

#include "op-attrs/operator_task_space_dim_idx_t.dtg.h"
#include "pcg/operator_space_to_machine_space_mapping.dtg.h"
#include "utils/orthotope/dim_domain_mapping.h"
#include "compiler/cost_estimator/one_to_one_communication_set.dtg.h"

namespace FlexFlow {

OneToOneCommunicationSet 
  one_to_one_communication_set_from_composition(
    OperatorSpaceToMachineSpaceMapping const &pre_map,
    DimDomainMapping<operator_task_space_dim_idx_t, operator_task_space_dim_idx_t> const &operator_task_space_map,
    OperatorSpaceToMachineSpaceMapping const &post_map);

} // namespace FlexFlow

#endif
