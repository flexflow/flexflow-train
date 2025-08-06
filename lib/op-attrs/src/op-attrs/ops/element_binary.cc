#include "op-attrs/ops/element_binary.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "utils/containers/require_same.h"
#include "utils/exception.h"

namespace FlexFlow {

TensorShape get_output_shape(ElementBinaryAttrs const &attrs,
                             TensorShape const &input_lhs,
                             TensorShape const &input_rhs) {
  assert(!(attrs.should_broadcast_lhs && attrs.should_broadcast_rhs));

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs, "Expected input shapes to match");

    return input_lhs;
  }
}

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &attrs,
                                     ParallelTensorShape const &input_lhs,
                                     ParallelTensorShape const &input_rhs) {
  assert(!(attrs.should_broadcast_lhs && attrs.should_broadcast_rhs));

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs, "Expected input shapes to match");

    switch (attrs.type) {
      case OperatorType::EW_ADD: {
        ASSERT(
            get_discard_copy_degree(input_lhs) == 1,
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

static void check_mapping_features(
    ElementBinaryAttrs const &attrs,
    num_ptensor_parallel_dims_t lhs_input_num_dims,
    num_ptensor_parallel_dims_t rhs_input_num_dims) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  ASSERT(lhs_input_num_dims == rhs_input_num_dims);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_lhs_input_mapping(
  ElementBinaryAttrs const &attrs,
  num_ptensor_parallel_dims_t lhs_input_num_dims,
  num_ptensor_parallel_dims_t rhs_input_num_dims) {

  check_mapping_features(attrs, lhs_input_num_dims, rhs_input_num_dims);

  return get_identity_mapping(lhs_input_num_dims);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_rhs_input_mapping(
  ElementBinaryAttrs const &attrs,
  num_ptensor_parallel_dims_t lhs_input_num_dims,
  num_ptensor_parallel_dims_t rhs_input_num_dims) {

  check_mapping_features(attrs, lhs_input_num_dims, rhs_input_num_dims);

  return get_identity_mapping(rhs_input_num_dims);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
  ElementBinaryAttrs const &attrs,
  num_ptensor_parallel_dims_t lhs_input_num_dims,
  num_ptensor_parallel_dims_t rhs_input_num_dims) {

  check_mapping_features(attrs, lhs_input_num_dims, rhs_input_num_dims);

  return get_identity_mapping(require_same(lhs_input_num_dims, rhs_input_num_dims));
}

} // namespace FlexFlow
