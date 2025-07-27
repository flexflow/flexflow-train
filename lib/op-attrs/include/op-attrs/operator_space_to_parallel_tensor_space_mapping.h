#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPING_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceMapping
  get_identity_mapping(nonnegative_int num_shard_dims);

} // namespace FlexFlow

#endif
