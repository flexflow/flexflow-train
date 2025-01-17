#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_PARALLEL_TENSOR_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_PARALLEL_TENSOR_SPACE_MAPPING_H

#include "op-attrs/operator_space_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/tensor_num_dims.dtg.h"

namespace FlexFlow {

OperatorSpaceParallelTensorSpaceMapping
  get_identity_mapping(TensorNumDims const &);

} // namespace FlexFlow

#endif
