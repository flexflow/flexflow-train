#include "local-pcg-execution/runtime_atomic_task_shard_binding.h"
#include "compiler/operator_atomic_task_shard_binding.h"
#include "op-attrs/tensor_role.dtg.h"
#include "task-spec/fwb_tensor_type.dtg.h"
#include "task-spec/symbolic_layer_training_tensor_group_signature.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

static std::unordered_map<symbolic_training_tensor_guid_t, ParallelTensorSpaceCoordinate> 
  get_tensor_shard_binding_for_type(
    SymbolicLayerTrainingTensorGroupSignature const &signature,
    OperatorAtomicTaskShardBinding const &shard_binding,
    TensorRole tensor_role,
    FwbTensorType tensor_type) {
  
  std::vector<symbolic_training_tensor_guid_t> keys 
    = get_training_tensors_for_role_and_type(signature, tensor_role, tensor_type);

  std::vector<ParallelTensorSpaceCoordinate> pt_coords
    = ptensor_space_coords_for_role(shard_binding, tensor_role);

  return map_from_keys_and_values(
    /*keys=*/keys,
    /*values=*/pt_coords);
};

RuntimeAtomicTaskShardBinding
  lower_op_shard_binding_to_runtime_shard_binding(OperatorAtomicTaskShardBinding const &op_shard_binding,
                                                  SymbolicLayerTrainingTensorGroupSignature const &signature) {

  auto get_bindings = [&](TensorRole tensor_role, FwbTensorType tensor_type) {
    return get_tensor_shard_binding_for_type(signature, op_shard_binding, tensor_role, tensor_type);
  };

  return RuntimeAtomicTaskShardBinding{
    merge_disjoint_maps(std::vector{
      get_bindings(TensorRole::INPUT, FwbTensorType::FORWARD),
      get_bindings(TensorRole::WEIGHT, FwbTensorType::FORWARD),
      get_bindings(TensorRole::OUTPUT, FwbTensorType::FORWARD),
    }),
  };
}

RuntimeAtomicTaskShardBinding 
  lower_op_shard_binding_to_bwd_pass_runtime_shard_binding(OperatorAtomicTaskShardBinding const &op_shard_binding,
                                                           SymbolicLayerTrainingTensorGroupSignature const &signature) {

  auto get_bindings = [&](TensorRole tensor_role, FwbTensorType tensor_type) {
    return get_tensor_shard_binding_for_type(signature, op_shard_binding, tensor_role, tensor_type);
  };

  return RuntimeAtomicTaskShardBinding{
    merge_disjoint_maps(std::vector{
      get_bindings(TensorRole::INPUT, FwbTensorType::FORWARD),
      get_bindings(TensorRole::WEIGHT, FwbTensorType::FORWARD),
      get_bindings(TensorRole::OUTPUT, FwbTensorType::FORWARD),
      get_bindings(TensorRole::INPUT, FwbTensorType::GRADIENT),
      get_bindings(TensorRole::WEIGHT, FwbTensorType::GRADIENT),
      get_bindings(TensorRole::OUTPUT, FwbTensorType::GRADIENT),
    }),
  };
}

RuntimeAtomicTaskShardBinding 
  lower_op_shard_binding_to_runtime_shard_binding(OperatorAtomicTaskShardBinding const &shard_binding,
                                                  SymbolicLayerTrainingTensorGroupSignature const &signature,
                                                  FwbOpTaskType task_type) {
  switch (task_type) {
    case FwbOpTaskType::FWD:
      return lower_op_shard_binding_to_fwd_pass_runtime_shard_binding(shard_binding, signature);
    case FwbOpTaskType::BWD:
      return lower_op_shard_binding_to_bwd_pass_runtime_shard_binding(shard_binding, signature);
    default:
      PANIC("Unhandled FwbOpTaskType", task_type);
  }
}

} // namespace FlexFlow
