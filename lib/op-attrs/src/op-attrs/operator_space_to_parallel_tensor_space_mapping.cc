#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/num_ptensor_shard_dims_t.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/orthotope/minimal_dim_domain_mapping.h"

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceMapping 
  empty_operator_space_to_ptensor_space_map() {

  return OperatorSpaceToParallelTensorSpaceMapping{
    empty_dim_domain_mapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>(),
  };
}

OperatorTaskSpace
  get_operator_task_space_for_mapping(OperatorSpaceToParallelTensorSpaceMapping const &mapping) {

  return operator_task_space_from_minimal_dim_domain(
    require_dim_domain_is_minimal(mapping.raw_mapping.l_domain));
}

ParallelTensorDimDegrees
  get_parallel_tensor_space_for_mapping(OperatorSpaceToParallelTensorSpaceMapping const &mapping) {

  return parallel_tensor_dim_degrees_from_dim_domain(mapping.raw_mapping.r_domain);
}


OperatorSpaceToParallelTensorSpaceMapping
    get_identity_mapping(
      OperatorTaskSpace const &operator_task_space,
      ParallelTensorDimDegrees const &parallel_tensor_dim_degrees) {

  MinimalDimDomain<parallel_tensor_dim_idx_t> pt_minimal_dim_domain 
    = minimal_dim_domain_from_parallel_tensor_dim_degrees(parallel_tensor_dim_degrees);

  ASSERT(op_task_space_num_dims(operator_task_space) == minimal_dim_domain_num_dims(pt_minimal_dim_domain));

  std::vector<operator_task_space_dim_idx_t> op_minimal_domain_dims
    = sorted_by(operator_task_space_get_dim_idxs(operator_task_space),
                get_operator_task_space_dim_ordering().lt);

  std::vector<parallel_tensor_dim_idx_t> pt_minimal_domain_dims 
    = sorted_by(get_minimal_domain_dims(pt_minimal_dim_domain), 
                get_parallel_tensor_dim_ordering().lt);

  bidict<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
    projection = bidict_from_keys_and_values(op_minimal_domain_dims, pt_minimal_domain_dims);

  return operator_ptensor_space_mapping_from_projection(
    DimProjection{EqProjection{projection}}, 
    operator_task_space,
    parallel_tensor_dim_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    operator_ptensor_space_mapping_from_projection(
      DimProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> const &projection,
      OperatorTaskSpace const &operator_task_space,
      ParallelTensorDimDegrees const &parallel_tensor_dim_degrees) {

  return OperatorSpaceToParallelTensorSpaceMapping{
    dim_domain_mapping_from_projection( 
      /*projection=*/projection,
      /*l_domain=*/lift_minimal_dim_domain(minimal_dim_domain_from_operator_task_space(operator_task_space)),
      /*r_domain=*/lift_minimal_dim_domain(minimal_dim_domain_from_parallel_tensor_dim_degrees(parallel_tensor_dim_degrees)),
      /*l_dim_ordering=*/get_operator_task_space_dim_ordering(),
      /*r_dim_ordering=*/get_parallel_tensor_dim_ordering()),
  };
}

OperatorSpaceToParallelTensorSpaceMapping
  operator_ptensor_space_mapping_from_composition(
    OperatorSpaceToParallelTensorSpaceMapping const &op_to_pt1_mapping,
    ParallelTensorSpaceToParallelTensorSpaceMapping const &pt1_to_pt2_mapping) {
  
  return OperatorSpaceToParallelTensorSpaceMapping{
    compose_dim_domain_mappings_through_minimal(
      op_to_pt1_mapping.raw_mapping,
      pt1_to_pt2_mapping.raw_mapping),
  };
}


ParallelTensorSpaceCoordinate
  ptensor_coord_for_task_space_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    TaskSpaceCoordinate const &task_space_coordinate) {

  DimCoord<parallel_tensor_dim_idx_t> dim_coord = 
    mapping.raw_mapping.at_l(
      dim_coord_from_task_space_coordinate(task_space_coordinate));

  return parallel_tensor_space_coord_from_dim_coord(dim_coord);
}

TaskSpaceCoordinate
  task_space_coord_for_ptensor_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    ParallelTensorSpaceCoordinate const &ptensor_space_coord) {
  
  DimCoord<operator_task_space_dim_idx_t> dim_coord = 
    mapping.raw_mapping.at_r(
      dim_coord_from_parallel_tensor_space_coord(ptensor_space_coord));

  return task_space_coordinate_from_dim_coord(dim_coord);
}


} // namespace FlexFlow
