#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "utils/containers/at_idx.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

ParallelTensorSpaceCoordinate
  ptensor_space_coord_for_slot_name(OperatorAtomicTaskShardBinding const &op_task_signature,
                                    TensorSlotName const &slot_name) {
  return op_task_signature.tensor_coords.at(slot_name);
}

} // namespace FlexFlow
