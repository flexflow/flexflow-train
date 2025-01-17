#include "op-attrs/operator_space_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/range.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"

namespace FlexFlow {

OperatorSpaceParallelTensorSpaceMapping
  get_identity_mapping(TensorNumDims const &tensor_num_dims) {
  
  std::set<parallel_tensor_dim_idx_t> parallel_tensor_dim_indices 
    = get_nontrivial_parallel_tensor_dim_indices(degrees);

  std::set<operator_task_space_dim_idx_t> operator_space_dim_indices
    = transform(set_of(range(parallel_tensor_dim_indices.size())), 
                [](int raw_idx) { return operator_task_space_dim_idx_t{raw_idx}; });

  bidict<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> raw_bidict
    = bidict_from_keys_and_values(vector_of(operator_space_dim_indices), vector_of(parallel_tensor_dim_indices));
 
  return OperatorSpaceParallelTensorSpaceMapping{DimProjection{EqProjection{raw_bidict}}};
}

} // namespace FlexFlow
