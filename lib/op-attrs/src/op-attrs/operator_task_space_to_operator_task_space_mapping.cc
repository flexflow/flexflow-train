#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

OperatorTaskSpace
  op_mapping_get_src_space(OperatorTaskSpaceToOperatorTaskSpaceMapping const &mapping) {
  
  return operator_task_space_from_minimal_dim_domain(
        require_dim_domain_is_minimal(mapping.raw_mapping.l_domain));
}

OperatorTaskSpace
  op_mapping_get_dst_space(OperatorTaskSpaceToOperatorTaskSpaceMapping const &mapping) {
  
  return operator_task_space_from_minimal_dim_domain(
        require_dim_domain_is_minimal(mapping.raw_mapping.r_domain));
}

bidict<TaskSpaceCoordinate, TaskSpaceCoordinate>
  op_to_op_get_coord_mapping(OperatorTaskSpaceToOperatorTaskSpaceMapping const &mapping) {
  return transform_values(
    transform_keys(mapping.raw_mapping.coord_mapping, 
                   task_space_coordinate_from_dim_coord), 
    task_space_coordinate_from_dim_coord);
}

OperatorTaskSpaceToOperatorTaskSpaceMapping 
  op_to_op_mapping_from_composition_through_tensor(
    OperatorSpaceToParallelTensorSpaceMapping const &src_to_tensor_mapping,
    OperatorSpaceToParallelTensorSpaceMapping const &dst_to_tensor_mapping) {
  return OperatorTaskSpaceToOperatorTaskSpaceMapping{
    compose_dim_domain_mappings(
      src_to_tensor_mapping.raw_mapping,
      invert_dim_domain_mapping(dst_to_tensor_mapping.raw_mapping)),
  };
}


} // namespace FlexFlow
