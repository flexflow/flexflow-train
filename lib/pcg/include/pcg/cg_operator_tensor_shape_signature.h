#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_CG_OPERATOR_TENSOR_SHAPE_SIGNATURE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_CG_OPERATOR_TENSOR_SHAPE_SIGNATURE_H

#include "pcg/cg_operator_tensor_shape_signature.dtg.h"
#include "pcg/tensor_role.dtg.h"

namespace FlexFlow {

std::vector<TensorShape> tensor_shapes_for_role(CGOperatorTensorShapeSignature const &signature,
                                                TensorRole tensor_role);

TensorShape tensor_shape_for_role_and_index(CGOperatorTensorShapeSignature const &signature,
                                            TensorRole tensor_role,
                                            nonnegative_int index);

} // namespace FlexFlow

#endif
