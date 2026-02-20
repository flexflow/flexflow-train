#include "compiler/cost_estimator/parallel_tensor_space_to_machine_space_mapping.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

ParallelTensorSpaceToMachineSpaceMapping ptensor_machine_map_from_composition(
    OperatorSpaceToMachineSpaceMapping const &op_task_to_machine_space_mapping,
    OperatorSpaceToParallelTensorSpaceMapping const
        &op_task_to_parallel_tensor_space_mapping) {
  ASSERT(op_task_to_machine_space_mapping.operator_task_space ==
         get_operator_task_space_for_mapping(
             op_task_to_parallel_tensor_space_mapping));

  bidict<ParallelTensorSpaceCoordinate, TaskSpaceCoordinate>
      pt_to_op_coord_map = transform_keys(
          transform_values(op_task_to_parallel_tensor_space_mapping.raw_mapping
                               .coord_mapping.reversed(),
                           task_space_coordinate_from_dim_coord),
          parallel_tensor_space_coord_from_dim_coord);

  bidict<TaskSpaceCoordinate, MachineSpaceCoordinate> op_to_ms_coord_map =
      op_task_to_machine_space_mapping.raw_mapping;

  return ParallelTensorSpaceToMachineSpaceMapping{
      /*raw_mapping=*/exhaustive_relational_join(pt_to_op_coord_map,
                                                 op_to_ms_coord_map),
      /*parallel_tensor_space=*/
      parallel_tensor_dim_degrees_from_dim_domain(
          op_task_to_parallel_tensor_space_mapping.raw_mapping.r_domain),
  };
};

} // namespace FlexFlow
