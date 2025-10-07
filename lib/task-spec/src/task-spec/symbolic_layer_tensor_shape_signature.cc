#include "task-spec/symbolic_layer_tensor_shape_signature.h"
#include "utils/containers/at_idx.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::vector<TensorShape>
  tensor_shapes_for_role(SymbolicLayerTensorShapeSignature const &signature,
                         TensorRole tensor_role) {
  switch (tensor_role) {
    case TensorRole::INPUT:
      return signature.input_shapes;
    case TensorRole::WEIGHT:
      return signature.weight_shapes;
    case TensorRole::OUTPUT:
      return signature.output_shapes;
    default:
      PANIC("Unhandled TensorRole", tensor_role);
  }
}

TensorShape
  tensor_shape_for_role_and_index(SymbolicLayerTensorShapeSignature const &signature,
                                  TensorRole tensor_role,
                                  nonnegative_int tensor_idx) {
  return at_idx(tensor_shapes_for_role(signature, tensor_role), tensor_idx);
}


} // namespace FlexFlow
