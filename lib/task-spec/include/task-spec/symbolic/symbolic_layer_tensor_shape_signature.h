#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_LAYER_TENSOR_SHAPE_SIGNATURE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_LAYER_TENSOR_SHAPE_SIGNATURE_H

#include "op-attrs/tensor_role.dtg.h"
#include "task-spec/symbolic/symbolic_layer_tensor_shape_signature.dtg.h"

namespace FlexFlow {

std::vector<TensorShape>
  tensor_shapes_for_role(SymbolicLayerTensorShapeSignature const &signaturte,
                         TensorRole tensor_role);

TensorShape
  tensor_shape_for_role_and_index(SymbolicLayerTensorShapeSignature const &signature,
                                  TensorRole tensor_role,
                                  nonnegative_int tensor_idx);

} // namespace FlexFlow

#endif
