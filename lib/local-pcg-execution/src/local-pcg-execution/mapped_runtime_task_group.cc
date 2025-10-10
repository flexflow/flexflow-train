#include "local-pcg-execution/mapped_runtime_task_group.h"
#include "compiler/operator_atomic_task_shard_binding.h"
#include "compiler/task_signature_tensor_key.h"
#include "local-pcg-execution/runtime_atomic_task_shard_binding.dtg.h"
#include "local-pcg-execution/runtime_atomic_task_shard_binding.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "pcg/machine_view.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/require_all_same.h"
#include "compiler/task_signature_tensor_key.dtg.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

MappedRuntimeTaskGroup::MappedRuntimeTaskGroup(
   bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &shard_bindings) 
  : shard_bindings(shard_bindings)
{
  auto check_arity = [&](TensorRole tensor_role) -> nonnegative_int {
    std::unordered_set<nonnegative_int> arities = 
      transform(shard_bindings.right_values(), 
                [&](OperatorAtomicTaskShardBinding const &s) -> nonnegative_int {
                  return num_elements(ptensor_space_coords_for_role(s, tensor_role));
                });

    return require_all_same(arities).value_or(0_n);
  };

  nonnegative_int num_inputs = check_arity(TensorRole::INPUT);
  nonnegative_int num_weights = check_arity(TensorRole::WEIGHT);
  nonnegative_int num_outputs = check_arity(TensorRole::OUTPUT);

  std::unordered_set<TaskSignatureTensorKey> all_keys =  
        all_keys_for_signature_arities(
          /*num_inputs=*/num_inputs,
          /*num_weights=*/num_weights,
          /*num_outputs=*/num_outputs);
          
  for (TaskSignatureTensorKey const &key : all_keys) {
    std::vector<OperatorAtomicTaskShardBinding> signatures_for_key = vector_of(shard_bindings.right_values());

    std::vector<ParallelTensorSpaceCoordinate> coords_for_key = 
      transform(signatures_for_key,
                [&](OperatorAtomicTaskShardBinding const &signature) {
                  return ptensor_space_coord_for_key(signature, key);
                });

    ASSERT(are_all_distinct(coords_for_key));

    std::vector<num_ptensor_parallel_dims_t> coord_dims_for_key = 
      transform(coords_for_key,
                [](ParallelTensorSpaceCoordinate const &c) {
                  return ptensor_coord_num_dims(c);
                });

    require_all_same(coord_dims_for_key);
  }
} 

bool MappedRuntimeTaskGroup::operator==(MappedRuntimeTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedRuntimeTaskGroup::operator!=(MappedRuntimeTaskGroup const &other) const {
  return this->tie() == other.tie();
}

std::tuple<
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &
> MappedRuntimeTaskGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &MappedRuntimeTaskGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

std::string format_as(::FlexFlow::MappedRuntimeTaskGroup const &m) {
  return fmt::format("<MappedRuntimeTaskGroup shard_bindings={}>", m.get_shard_bindings());
}

std::ostream &operator<<(std::ostream &s, ::FlexFlow::MappedRuntimeTaskGroup const &x) {
  return (s << fmt::to_string(x));
}

MappedRuntimeTaskGroup
  lower_mapped_operator_task_group_to_mapped_runtime_task_group(MappedOperatorTaskGroup const &op_task_group,
                                                                SymbolicLayerTrainingTensorGroupSignature const &symbolic_layer_signature, 
                                                                FwbOpTaskType task_type) {
  return MappedRuntimeTaskGroup{
    transform_values(
      op_task_group.get_shard_bindings(),
      [&](OperatorAtomicTaskShardBinding const &op_shard_binding) 
        -> RuntimeAtomicTaskShardBinding
      {
        return lower_op_shard_binding_to_runtime_shard_binding(op_shard_binding, symbolic_layer_signature, task_type);
      }),
  };
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedRuntimeTaskGroup>::operator()(::FlexFlow::MappedRuntimeTaskGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std
