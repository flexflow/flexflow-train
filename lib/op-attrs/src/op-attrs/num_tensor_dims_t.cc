#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/num_ptensor_shard_dims_t.h"

namespace FlexFlow {

num_tensor_dims_t num_tensor_dims_from_num_ptensor_shard_dims(num_ptensor_shard_dims_t num_ptensor_shard_dims) {
  return num_tensor_dims_t{num_ptensor_shard_dims.value};
}

num_tensor_dims_t num_tensor_dims_from_num_ptensor_parallel_dims(num_ptensor_parallel_dims_t num_ptensor_parallel_dims) {
  return num_tensor_dims_from_num_ptensor_shard_dims(
    num_ptensor_shard_dims_from_parallel_dims(
      num_ptensor_parallel_dims));
}

num_ptensor_shard_dims_t num_ptensor_shard_dims_from_num_tensor_dims(num_tensor_dims_t num_tensor_dims) {
  return num_ptensor_shard_dims_t{num_tensor_dims.value};
}

num_ptensor_parallel_dims_t num_ptensor_parallel_dims_from_num_tensor_dims(num_tensor_dims_t num_tensor_dims) {
  return num_ptensor_parallel_dims_from_shard_dims(
    num_ptensor_shard_dims_from_num_tensor_dims(
      num_tensor_dims));
}



} // namespace FlexFlow
