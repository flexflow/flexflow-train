#include "op-attrs/num_ptensor_shard_dims_t.h"

namespace FlexFlow {

num_ptensor_parallel_dims_t num_ptensor_parallel_dims_from_shard_dims(
    num_ptensor_shard_dims_t num_shard_dims) {
  return num_ptensor_parallel_dims_t{num_shard_dims.value + 2_p};
}

num_ptensor_shard_dims_t num_ptensor_shard_dims_from_parallel_dims(
    num_ptensor_parallel_dims_t num_parallel_dims) {
  return num_ptensor_shard_dims_t{
      nonnegative_int{num_parallel_dims.int_from_num_ptensor_parallel_dims() -
                      2},
  };
}

} // namespace FlexFlow
