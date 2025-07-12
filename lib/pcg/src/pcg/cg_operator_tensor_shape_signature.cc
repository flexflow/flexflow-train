#include "pcg/cg_operator_tensor_shape_signature.h"

namespace FlexFlow {

std::vector<TensorShape>
    tensor_shapes_for_role(CGOperatorTensorShapeSignature const &signature,
                           TensorRole tensor_role) {
  switch (tensor_role) {
    case TensorRole::INPUT:
      return signature.input_shapes;
    case TensorRole::WEIGHT:
      return signature.weight_shapes;
    case TensorRole::OUTPUT:
      return signature.output_shapes;
    default:
      PANIC("Unhandled tensor role", tensor_role);
  };
}

TensorShape tensor_shape_for_role_and_index(
    CGOperatorTensorShapeSignature const &signature,
    TensorRole tensor_role,
    nonnegative_int index) {
  return tensor_shapes_for_role(signature, tensor_role)
      .at(index.unwrap_nonnegative());
}

} // namespace FlexFlow
