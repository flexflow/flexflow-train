#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PARALLEL_TENSOR_SPACE_TO_MACHINE_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_PARALLEL_TENSOR_SPACE_TO_MACHINE_SPACE_MAPPING_H

#include "compiler/cost_estimator/parallel_tensor_space_to_machine_space_mapping.dtg.h"
#include "pcg/operator_space_to_machine_space_mapping.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"

namespace FlexFlow {

ParallelTensorSpaceToMachineSpaceMapping
  ptensor_machine_map_from_composition(
    OperatorSpaceToMachineSpaceMapping const &op_task_to_machine_space_mapping, 
    OperatorSpaceToParallelTensorSpaceMapping const &op_task_to_parallel);

} // namespace FlexFlow

#endif
