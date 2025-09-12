#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/orthotope/minimal_dim_domain_mapping.h"

namespace FlexFlow {

OperatorTaskSpaceToOperatorTaskSpaceMapping op_to_op_identity_mapping(
  OperatorTaskSpace const &src_space,
  OperatorTaskSpace const &dst_space) {

  return OperatorTaskSpaceToOperatorTaskSpaceMapping{
    dim_domain_mapping_identity_map(
      /*l_domain=*/lift_minimal_dim_domain(minimal_dim_domain_from_operator_task_space(src_space)),
      /*r_domain=*/lift_minimal_dim_domain(minimal_dim_domain_from_operator_task_space(dst_space)),
      /*l_dim_ordering=*/get_operator_task_space_dim_ordering(),
      /*r_dim_ordering=*/get_operator_task_space_dim_ordering()),
  };
}

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
  
  MinimalDimDomainMapping<
    operator_task_space_dim_idx_t,
    parallel_tensor_dim_idx_t
  > src_to_pt = minimal_mapping_from_dim_domain_mapping(src_to_tensor_mapping.raw_mapping);

  std::unordered_set<operator_task_space_dim_idx_t> src_trivial_dims 
    = get_trivial_domain_dims(src_to_tensor_mapping.raw_mapping.l_domain);

  MinimalDimDomainMapping<
    parallel_tensor_dim_idx_t,
    operator_task_space_dim_idx_t
  > pt_to_dst = minimal_mapping_from_dim_domain_mapping(invert_dim_domain_mapping(src_to_tensor_mapping.raw_mapping));

  std::unordered_set<operator_task_space_dim_idx_t> dst_trivial_dims 
    = get_trivial_domain_dims(dst_to_tensor_mapping.raw_mapping.l_domain);

  return OperatorTaskSpaceToOperatorTaskSpaceMapping{
    dim_domain_mapping_from_minimal_dim_domain(
      compose_minimal_dim_domain_mappings(src_to_pt, pt_to_dst), 
      src_trivial_dims, 
      dst_trivial_dims),
  };
}


} // namespace FlexFlow
