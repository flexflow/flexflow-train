#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceMapping
    get_identity_mapping(nonnegative_int num_shard_dims) {

  std::set<parallel_tensor_dim_idx_t> parallel_tensor_dim_indices =
      dim_idxs_for_num_shard_dims(num_shard_dims);

  std::set<operator_task_space_dim_idx_t> operator_space_dim_indices =
      transform(set_of(range(num_elements(parallel_tensor_dim_indices))),
                [](nonnegative_int raw_idx) {
                  return operator_task_space_dim_idx_t{raw_idx};
                });

  bidict<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> raw_bidict =
      bidict_from_keys_and_values(vector_of(operator_space_dim_indices),
                                  vector_of(parallel_tensor_dim_indices));

  return OperatorSpaceToParallelTensorSpaceMapping{
      DimProjection{EqProjection{raw_bidict}}};
}

} // namespace FlexFlow
