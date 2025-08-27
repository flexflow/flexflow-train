#include "compiler/cost_estimator/one_to_one_communication_set.h"
#include "op-attrs/operator_task_space.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

OneToOneCommunicationSet 
  one_to_one_communication_set_from_composition(
    OperatorSpaceToMachineSpaceMapping const &pre_map,
    DimDomainMapping<operator_task_space_dim_idx_t, operator_task_space_dim_idx_t> const &op_space_mapping,
    OperatorSpaceToMachineSpaceMapping const &post_map) {

  OperatorTaskSpace l_task_space = 
    operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(op_space_mapping.l_domain));

  OperatorTaskSpace r_task_space = 
    operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(op_space_mapping.r_domain));

  ASSERT(pre_map.operator_task_space == l_task_space);
  ASSERT(post_map.operator_task_space == r_task_space);

  bidict<MachineSpaceCoordinate, TaskSpaceCoordinate>
    raw_pre_map = pre_map.raw_mapping.reversed();

  bidict<TaskSpaceCoordinate, TaskSpaceCoordinate> 
    raw_op_space_mapping = 
      transform_keys(
        transform_values(
          op_space_mapping.coord_mapping,
          task_space_coordinate_from_dim_coord),
        task_space_coordinate_from_dim_coord);

  bidict<TaskSpaceCoordinate, MachineSpaceCoordinate> 
    raw_post_map = post_map.raw_mapping;

  return OneToOneCommunicationSet{
    /*raw_mapping=*/
      exhaustive_relational_join(
        exhaustive_relational_join(
          raw_pre_map,
          raw_op_space_mapping),
        raw_post_map),
  };
}


} // namespace FlexFlow
