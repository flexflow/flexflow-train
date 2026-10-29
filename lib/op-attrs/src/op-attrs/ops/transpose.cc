#include "op-attrs/ops/transpose.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/bidict/algorithms/bidict_transform_keys.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"
#include "utils/orthotope/dim_coord.h"

namespace FlexFlow {

TensorShape transpose_get_output_shape(TransposeAttrs const &attrs,
                                       TensorShape const &input_shape) {
  return permute_tensor_shape(attrs.permutation, input_shape);
}

ParallelTensorDimDegrees transpose_get_output_parallel_dim_degrees(
    TransposeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  return permute_parallel_tensor_dim_degrees(attrs.permutation, input_degrees);
}

ParallelTensorShape transpose_get_output_parallel_shape(
    TransposeAttrs const &attrs, ParallelTensorShape const &input_shape) {
  TensorShape output_shape =
      transpose_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      transpose_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

StandardOperatorTaskGroup
    transpose_get_task_group(TransposeAttrs const &attrs,
                             ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      transpose_get_output_parallel_dim_degrees(attrs, input_degrees);

  return StandardOperatorTaskGroup{
      transform(get_parallel_tensor_space_coordinates(input_degrees),
                [&](ParallelTensorSpaceCoordinate const &input_coord)
                    -> AbstractedOperatorAtomicTaskShardBinding {
                  ParallelTensorSpaceCoordinate output_coord =
                      permute_parallel_tensor_space_coordinate(
                          attrs.permutation, input_coord);

                  return AbstractedOperatorAtomicTaskShardBinding{
                      /*tensor_coords=*/{
                          {
                              TensorSlotName::INPUT,
                              input_coord,
                          },
                          {
                              TensorSlotName::OUTPUT,
                              output_coord,
                          },
                      },
                      /*task_coord=*/
                      task_coord_matching_parallel_tensor_space_coordinate(
                          output_coord, output_degrees),
                  };
                }),
  };
}

} // namespace FlexFlow
