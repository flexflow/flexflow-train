#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "utils/containers/at_idx.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::vector<ParallelTensorSpaceCoordinate>
  ptensor_space_coords_for_role(OperatorAtomicTaskShardBinding const &op_task_signature,
                                TensorRole tensor_role) {
  switch (tensor_role) {
    case TensorRole::INPUT:
      return op_task_signature.inputs;
    case TensorRole::WEIGHT:
      return op_task_signature.weights;
    case TensorRole::OUTPUT:
      return op_task_signature.outputs;
    default: 
      PANIC("Unhandled TensorRole", tensor_role); 
  };
}

ParallelTensorSpaceCoordinate
  ptensor_space_coord_for_key(OperatorAtomicTaskShardBinding const &op_task_signature,
                              TaskSignatureTensorKey const &tensor_key) {
  return at_idx(
    ptensor_space_coords_for_role(op_task_signature, tensor_key.tensor_role),
    tensor_key.idx);
}

} // namespace FlexFlow
