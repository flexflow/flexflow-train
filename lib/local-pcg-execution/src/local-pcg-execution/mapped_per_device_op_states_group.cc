#include "local-pcg-execution/mapped_per_device_op_states_group.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "compiler/machine_mapping/machine_view.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

MappedPerDeviceOpStatesGroup::MappedPerDeviceOpStatesGroup(
   bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &per_device_op_states) 
  : per_device_op_states(per_device_op_states)
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

bool MappedPerDeviceOpStatesGroup::operator==(MappedPerDeviceOpStatesGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedPerDeviceOpStatesGroup::operator!=(MappedPerDeviceOpStatesGroup const &other) const {
  return this->tie() == other.tie();
}

std::tuple<
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &
> MappedPerDeviceOpStatesGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &MappedPerDeviceOpStatesGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

std::string format_as(::FlexFlow::MappedPerDeviceOpStatesGroup const &m) {
  return fmt::format("<MappedPerDeviceOpStatesGroup shard_bindings={}>", m.get_shard_bindings());
}

std::ostream &operator<<(std::ostream &s, ::FlexFlow::MappedPerDeviceOpStatesGroup const &x) {
  return (s << fmt::to_string(x));
}

MappedPerDeviceOpStatesGroup
  mapped_operator_task_group_from_machine_view(
    ComputationGraphOpAttrs const &op_attrs,
    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const &inputs_dim_degrees,
    MachineView const &machine_view) {

  OperatorTaskSpace op_task_space = get_operator_task_space(op_attrs, inputs_dim_degrees);  

  return MappedPerDeviceOpStatesGroup{
    generate_bidict(get_machine_space_coordinates(op_task_space, machine_view),
                    [&](MachineSpaceCoordinate const &machine_space_coord) {
                      return operator_atomic_task_shard_binding_from_machine_view(
                        op_attrs, 
                        inputs_dim_degrees,
                        machine_view,
                        machine_space_coord);
                    }),
  };
}


} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedPerDeviceOpStatesGroup>::operator()(::FlexFlow::MappedPerDeviceOpStatesGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std
