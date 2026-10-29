#include "op-attrs/ops/element_unary.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/orthotope/minimal_dim_domain.h"

namespace FlexFlow {

ElementUnaryAttrs make_relu_attrs() {
  return ElementUnaryAttrs{
      /*op_type=*/ElementUnaryOp::RELU,
      /*scalar=*/std::nullopt,
  };
}

TensorShape element_unary_get_output_shape(ElementUnaryAttrs const &attrs,
                                           TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorShape element_unary_get_output_parallel_shape(
    ElementUnaryAttrs const &attrs, ParallelTensorShape const &input_shape) {
  TensorShape output_shape =
      element_unary_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      element_unary_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorDimDegrees element_unary_get_output_parallel_dim_degrees(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ASSERT(input_degrees.sum_degree.value == 1);

  return input_degrees;
}

StandardOperatorTaskGroup element_unary_get_task_group(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      element_unary_get_output_parallel_dim_degrees(attrs, input_degrees);

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
