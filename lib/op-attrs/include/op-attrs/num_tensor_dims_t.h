#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_NUM_TENSOR_DIMS_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_NUM_TENSOR_DIMS_T_H

#include "op-attrs/num_tensor_dims_t.dtg.h"
#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"

namespace FlexFlow {

num_tensor_dims_t 
  num_tensor_dims_from_num_ptensor_shard_dims(num_ptensor_shard_dims_t);

num_tensor_dims_t num_tensor_dims_from_num_ptensor_parallel_dims(num_ptensor_parallel_dims_t);

num_ptensor_shard_dims_t num_ptensor_shard_dims_from_num_tensor_dims(num_tensor_dims_t);

num_ptensor_parallel_dims_t num_ptensor_parallel_dims_from_num_tensor_dims(num_tensor_dims_t);

} // namespace FlexFlow

#endif
