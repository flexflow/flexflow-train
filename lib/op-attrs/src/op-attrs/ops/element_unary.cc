#include "op-attrs/ops/element_unary.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

ElementUnaryAttrs make_relu_attrs() {
  return ElementUnaryAttrs{
      /*op_type=*/OperatorType::RELU,
      /*scalar=*/std::nullopt,
  };
}

tl::expected<TensorShape, std::string>
    get_output_shape(ElementUnaryAttrs const &attrs,
                     TensorShape const &input_shape) {
  return input_shape;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ElementUnaryAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  if (get_sum_degree(input_shape) != 1) {
    return tl::unexpected(
        fmt::format("Expected sum degree 1, but receieved sum degree {}",
                    get_sum_degree(input_shape)));
  }

  if (get_discard_copy_degree(input_shape) != 1) {
    return tl::unexpected(fmt::format(
        "Expected discard copy degree 1, but received discartd copy degree {}",
        get_discard_copy_degree(input_shape)));
  }

  return input_shape;
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    ElementUnaryAttrs const &attrs,
    num_ptensor_parallel_dims_t input_num_dims) {
  return get_identity_mapping(input_num_dims);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    ElementUnaryAttrs const &attrs,
    num_ptensor_parallel_dims_t input_num_dims) {
  return get_identity_mapping(input_num_dims);
}

} // namespace FlexFlow
