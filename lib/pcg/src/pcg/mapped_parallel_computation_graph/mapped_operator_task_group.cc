#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "pcg/mapped_parallel_computation_graph/task_signature_tensor_key.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/require_all_same.h"
#include "pcg/mapped_parallel_computation_graph/task_signature_tensor_key.dtg.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

MappedOperatorTaskGroup::MappedOperatorTaskGroup(
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

bool MappedOperatorTaskGroup::operator==(MappedOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedOperatorTaskGroup::operator!=(MappedOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

std::tuple<
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &
> MappedOperatorTaskGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &MappedOperatorTaskGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

std::string format_as(::FlexFlow::MappedOperatorTaskGroup const &m) {
  return fmt::format("<MappedOperatorTaskGroup shard_bindings={}>", m.get_shard_bindings());
}

std::ostream &operator<<(std::ostream &s, ::FlexFlow::MappedOperatorTaskGroup const &x) {
  return (s << fmt::to_string(x));
}


} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedOperatorTaskGroup>::operator()(::FlexFlow::MappedOperatorTaskGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std
