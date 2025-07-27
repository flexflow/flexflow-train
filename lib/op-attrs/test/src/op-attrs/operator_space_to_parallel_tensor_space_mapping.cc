#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow; 

static parallel_tensor_dim_idx_t shard_dim_idx_from_raw(int idx) {
  return parallel_tensor_dim_idx_t{ff_dim_t{nonnegative_int{idx}}};
}

static operator_task_space_dim_idx_t op_task_space_dim_from_raw(int idx) {
  return operator_task_space_dim_idx_t{nonnegative_int{idx}};
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_identity_mapping(ParallelTensorDimDegrees)") {
    nonnegative_int num_shard_dims = nonnegative_int{2};

    OperatorSpaceToParallelTensorSpaceMapping result = get_identity_mapping(num_shard_dims);

    NOT_IMPLEMENTED();
    // CHECK(result == correct);
  }
}
