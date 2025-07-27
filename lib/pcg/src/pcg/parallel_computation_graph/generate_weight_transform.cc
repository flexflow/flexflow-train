#include "pcg/parallel_computation_graph/generate_weight_transform.h"
#include "op-attrs/ff_ordered/enumerate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::unordered_set<ParallelOpAttrs>
    generate_weight_transform(TensorShape const &current,
                              ParallelTensorShape const &goal) {
  std::unordered_set<ParallelOpAttrs> result;

  positive_int sum_degree = get_sum_degree(goal);
  ASSERT(sum_degree == 1, 
         "generate_weight_transform currently only supports sum_degree = 1");

  positive_int discard_copy_degree = get_discard_copy_degree(goal);
  if (discard_copy_degree != 1) {
    result.insert(ParallelOpAttrs{ReplicateAttrs{discard_copy_degree}});
  }

  for (auto const &[shard_dim, shard_degree] :
       enumerate(ff_ordered_shard_degrees(goal))) {
    if (shard_degree != 1) {
      result.insert(ParallelOpAttrs{RepartitionAttrs{shard_dim, shard_degree}});
    }
  }

  return result;
}

} // namespace FlexFlow
