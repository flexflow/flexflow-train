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

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceMapping
    get_identity_mapping(num_ptensor_parallel_dims_t total_num_parallel_dims) {

  auto mapping = make_empty_eq_projection< 
    operator_task_space_dim_idx_t,
    parallel_tensor_dim_idx_t
  >();

  nonnegative_int op_space_dim_idx = 0_n;
  auto project_dim = [&](parallel_tensor_dim_idx_t ptensor_dim_idx) {
    project_dims(mapping, 
                 operator_task_space_dim_idx_t{op_space_dim_idx},
                 ptensor_dim_idx);
    op_space_dim_idx++;
  };

  project_dim(sum_dim_idx());
  project_dim(discard_copy_dim_idx());

  nonnegative_int num_shard_dims = num_ptensor_shard_dims_from_parallel_dims(total_num_parallel_dims).value;
  for (nonnegative_int shard_dim : nonnegative_range(num_shard_dims)) {
    project_dim(shard_dim_idx(ff_dim_t{shard_dim}));
  }

  return OperatorSpaceToParallelTensorSpaceMapping{DimProjection{mapping}};
}

OperatorSpaceToParallelTensorSpaceMapping
  oerator_ptensor_space_mapping_from_composition(
    OperatorSpaceToParallelTensorSpaceMapping const &op_to_pt1_mapping,
    ParallelTensorSpaceMapping const &pt1_to_pt2_mapping) {

  DimProjection<
    operator_task_space_dim_idx_t, 
    parallel_tensor_dim_idx_t
  > op_to_pt1 = op_to_pt1_mapping.raw_projection;

  DimProjection<
    parallel_tensor_dim_idx_t,
    parallel_tensor_dim_idx_t
  > pt1_to_pt2 = pt1_to_pt2_mapping.raw_projection;

  DimProjection<
    operator_task_space_dim_idx_t,
    parallel_tensor_dim_idx_t
  > op_to_pt2 = compose_dim_projections(op_to_pt1, pt1_to_pt2);

  return OperatorSpaceToParallelTensorSpaceMapping{
    op_to_pt2,
  };
}


ParallelTensorSpaceCoordinate
  ptensor_coord_for_task_space_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    OperatorTaskSpace const &op_task_space,
    ParallelTensorDimDegrees const &ptensor_dim_degrees,
    TaskSpaceCoordinate const &task_space_coordinate) {

  DimCoord<parallel_tensor_dim_idx_t> dim_coord = 
    compute_projection(
      /*projection=*/mapping.raw_projection,
      /*input_coord=*/dim_coord_from_task_space_coordinate(task_space_coordinate),
      /*input_domain=*/dim_domain_from_operator_task_space(op_task_space),
      /*output_domain=*/dim_domain_from_parallel_tensor_dim_degrees(ptensor_dim_degrees),
      /*input_dim_ordering=*/get_operator_task_space_dim_ordering(),
      /*output_dim_ordering=*/get_parallel_tensor_dim_ordering());

  return parallel_tensor_space_coord_from_dim_coord(dim_coord);
}

TaskSpaceCoordinate
  task_space_coord_for_ptensor_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    ParallelTensorDimDegrees const &ptensor_dim_degrees,
    OperatorTaskSpace const &op_task_space,
    ParallelTensorSpaceCoordinate const &ptensor_space_coord) {

  DimCoord<operator_task_space_dim_idx_t> dim_coord = 
    compute_projection(
      /*projection=*/invert_dim_projection(mapping.raw_projection),
      /*input_coord=*/dim_coord_from_parallel_tensor_space_coord(ptensor_space_coord),
      /*input_domain=*/dim_domain_from_parallel_tensor_dim_degrees(ptensor_dim_degrees),
      /*output_domain=*/dim_domain_from_operator_task_space(op_task_space),
      /*input_dim_ordering=*/get_parallel_tensor_dim_ordering(),
      /*output_dim_ordering=*/get_operator_task_space_dim_ordering());

  return task_space_coordinate_from_dim_coord(dim_coord);
}


} // namespace FlexFlow
