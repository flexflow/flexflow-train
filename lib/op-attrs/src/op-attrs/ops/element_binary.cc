#include "op-attrs/ops/element_binary.h"
#include "utils/exception.h"

namespace FlexFlow {

TensorShape
    get_output_shape(ElementBinaryAttrs const &attrs,
                     TensorShape const &input_lhs,
                     TensorShape const &input_rhs) {
  assert(!(attrs.should_broadcast_lhs && attrs.should_broadcast_rhs));

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs,
           "Expected input shapes to match");

    return input_lhs;
  }
}

ParallelTensorShape
    get_output_shape(ElementBinaryAttrs const &attrs,
                     ParallelTensorShape const &input_lhs,
                     ParallelTensorShape const &input_rhs) {
  assert(!(attrs.should_broadcast_lhs && attrs.should_broadcast_rhs));

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs, 
           "Expected input shapes to match");

    switch (attrs.type) {
      case OperatorType::EW_ADD: {
        ASSERT(get_discard_copy_degree(input_lhs) == 1,
               "Elementwise Add expected discard copy degree of inputs to be 1");

        break;
      }
      case OperatorType::EW_SUB:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MUL:
        NOT_IMPLEMENTED();
      case OperatorType::EW_DIV:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MAX:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MIN:
        NOT_IMPLEMENTED();
      default:
        PANIC("Unexpected element-wise binary operator", attrs.type);
    }

    return input_lhs;
  }
}

} // namespace FlexFlow
