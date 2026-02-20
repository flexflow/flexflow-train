#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/orthotope/minimal_dim_domain_mapping.h"

namespace FlexFlow {

OperatorTaskSpaceToOperatorTaskSpaceMapping
    op_to_op_identity_mapping(OperatorTaskSpace const &src_space,
                              OperatorTaskSpace const &dst_space) {

  return OperatorTaskSpaceToOperatorTaskSpaceMapping{
      dim_domain_mapping_identity_map(
          /*l_domain=*/lift_minimal_dim_domain(
              minimal_dim_domain_from_operator_task_space(src_space)),
          /*r_domain=*/
          lift_minimal_dim_domain(
              minimal_dim_domain_from_operator_task_space(dst_space)),
          /*l_dim_ordering=*/get_operator_task_space_dim_ordering(),
          /*r_dim_ordering=*/get_operator_task_space_dim_ordering()),
  };
}

OperatorTaskSpace op_mapping_get_src_space(
    OperatorTaskSpaceToOperatorTaskSpaceMapping const &mapping) {

  return operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(mapping.raw_mapping.l_domain));
}

OperatorTaskSpace op_mapping_get_dst_space(
    OperatorTaskSpaceToOperatorTaskSpaceMapping const &mapping) {

  return operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(mapping.raw_mapping.r_domain));
}

bidict<TaskSpaceCoordinate, TaskSpaceCoordinate> op_to_op_get_coord_mapping(
    OperatorTaskSpaceToOperatorTaskSpaceMapping const &mapping) {
  return transform_values(transform_keys(mapping.raw_mapping.coord_mapping,
                                         task_space_coordinate_from_dim_coord),
                          task_space_coordinate_from_dim_coord);
}

OperatorTaskSpaceToOperatorTaskSpaceMapping
    op_to_op_mapping_from_composition_through_tensor(
        OperatorSpaceToParallelTensorSpaceMapping const &src_to_tensor_mapping,
        OperatorSpaceToParallelTensorSpaceMapping const
            &dst_to_tensor_mapping) {

  return OperatorTaskSpaceToOperatorTaskSpaceMapping{
      compose_dim_domain_mappings_through_minimal(
          src_to_tensor_mapping.raw_mapping,
          invert_dim_domain_mapping(dst_to_tensor_mapping.raw_mapping)),
  };
}

} // namespace FlexFlow
