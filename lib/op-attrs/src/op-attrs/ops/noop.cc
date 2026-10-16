#include "op-attrs/ops/noop.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

TensorShape noop_get_output_shape(NoopAttrs const &,
                                  TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorDimDegrees noop_get_output_parallel_dim_degrees(
    NoopAttrs const &, ParallelTensorDimDegrees const &input_dim_degrees) {
  return input_dim_degrees;
}

ParallelTensorShape
    noop_get_output_parallel_shape(NoopAttrs const &,
                                   ParallelTensorShape const &input_shape) {
  return input_shape;
}

StandardOperatorTaskGroup
    noop_get_task_group(NoopAttrs const &attrs,
                        ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      noop_get_output_parallel_dim_degrees(attrs, input_degrees);

  return StandardOperatorTaskGroup{
      transform(get_parallel_tensor_space_coordinates(input_degrees),
                [&](ParallelTensorSpaceCoordinate const &input_coord)
                    -> AbstractedOperatorAtomicTaskShardBinding {
                  ParallelTensorSpaceCoordinate output_coord = input_coord;

                  return AbstractedOperatorAtomicTaskShardBinding{
                      /*tensor_corods=*/{
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
