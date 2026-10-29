#include "op-attrs/ops/element_binary.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/containers/binary_cartesian_product.h"
#include "utils/containers/require_same.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape element_binary_get_output_shape(ElementBinaryAttrs const &attrs,
                                            TensorShape const &input_lhs,
                                            TensorShape const &input_rhs) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs, "Expected input shapes to match");

    return input_lhs;
  }
}

ParallelTensorShape element_binary_get_output_parallel_shape(
    ElementBinaryAttrs const &attrs,
    ParallelTensorShape const &input_lhs,
    ParallelTensorShape const &input_rhs) {
  TensorShape output_shape = element_binary_get_output_shape(
      attrs, get_reduced_shape(input_lhs), get_reduced_shape(input_rhs));

  ParallelTensorDimDegrees output_degrees =
      element_binary_get_output_parallel_dim_degrees(
          attrs,
          get_parallel_degrees(input_lhs),
          get_parallel_degrees(input_rhs));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorDimDegrees element_binary_get_output_parallel_dim_degrees(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  ASSERT(lhs_input_degrees == rhs_input_degrees);

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(lhs_input_degrees == rhs_input_degrees,
           "Expected input degrees to match");

    switch (attrs.op) {
      case ElementBinaryOp::ADD: {
        break;
      }
      case ElementBinaryOp::SUBTRACT:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::MULTIPLY:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::DIVIDE:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::MAX:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::MIN:
        NOT_IMPLEMENTED();
      default:
        PANIC("Unexpected element-wise binary operator", attrs.op);
    }

    return lhs_input_degrees;
  }
}

StandardOperatorTaskGroup element_binary_get_task_group(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  ParallelTensorDimDegrees output_degrees =
      element_binary_get_output_parallel_dim_degrees(
          attrs, lhs_input_degrees, rhs_input_degrees);

  return StandardOperatorTaskGroup{
      transform(get_parallel_tensor_space_coordinates(output_degrees),
                [&](ParallelTensorSpaceCoordinate const &coord)
                    -> AbstractedOperatorAtomicTaskShardBinding {
                  return AbstractedOperatorAtomicTaskShardBinding{
                      /*tensor_corods=*/{
                          {
                              TensorSlotName::LHS_INPUT,
                              coord,
                          },
                          {
                              TensorSlotName::RHS_INPUT,
                              coord,
                          },
                          {
                              TensorSlotName::OUTPUT,
                              coord,
                          },
                      },
                      /*task_coord=*/
                      task_coord_matching_parallel_tensor_space_coordinate(
                          coord, output_degrees),
                  };
                }),
  };
}

} // namespace FlexFlow
