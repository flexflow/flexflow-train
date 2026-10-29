#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/standard_operator_task_group.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "utils/binary_relation/binary_relation_is_biunique.h"
#include "utils/containers/get_one_of.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/set.h"
#include "utils/hash/tuple.h"
#include "utils/optional.h"

namespace FlexFlow {

ShardSignatureInstance::ShardSignatureInstance(
    std::set<OperatorAtomicTaskShardBinding> const &shard_bindings)
    : bindings(shard_bindings) {

  std::set<TensorSlotName> slot_names = [&]() -> std::set<TensorSlotName> {
    std::vector<std::set<TensorSlotName>> binding_slot_sets = transform(
        vector_of(this->bindings),
        [&](OperatorAtomicTaskShardBinding const &s)
            -> std::set<TensorSlotName> { return keys(s.tensor_coords); });

    return require_all_same1(binding_slot_sets);
  }();

  auto coords_for_slot = [&](TensorSlotName const &slot_name)
      -> std::set<ParallelTensorSpaceCoordinate> {
    return transform(this->bindings,
                     [&](OperatorAtomicTaskShardBinding const &b)
                         -> ParallelTensorSpaceCoordinate {
                       return b.tensor_coords.at(slot_name);
                     });
  };

  for (TensorSlotName const &slot_name : slot_names) {
    ASSERT(parallel_tensor_coord_set_is_orthotopic(coords_for_slot(slot_name)));
  }

  auto coord_relation_for_slots = [&](TensorSlotName slot_name1,
                                      TensorSlotName slot_name2)
      -> BinaryRelation<ParallelTensorSpaceCoordinate,
                        ParallelTensorSpaceCoordinate> {
    return BinaryRelation<ParallelTensorSpaceCoordinate,
                          ParallelTensorSpaceCoordinate>{
        transform(shard_bindings,
                  [&](OperatorAtomicTaskShardBinding const &b)
                      -> std::pair<ParallelTensorSpaceCoordinate,
                                   ParallelTensorSpaceCoordinate> {
                    return std::pair{
                        b.tensor_coords.at(slot_name1),
                        b.tensor_coords.at(slot_name2),
                    };
                  }),
    };
  };

  TensorSlotName chosen = get_one_of(slot_names);

  for (TensorSlotName const &slot_name : slot_names) {
    BinaryRelation<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate>
        coord_rel = coord_relation_for_slots(chosen, slot_name);
    ASSERT(binary_relation_is_biunique(coord_rel));
  }
}

bool ShardSignatureInstance::operator==(
    ShardSignatureInstance const &other) const {
  return this->tie() == other.tie();
}

bool ShardSignatureInstance::operator!=(
    ShardSignatureInstance const &other) const {
  return this->tie() != other.tie();
}

bool ShardSignatureInstance::operator<(
    ShardSignatureInstance const &other) const {
  return this->tie() < other.tie();
}

bool ShardSignatureInstance::operator>(
    ShardSignatureInstance const &other) const {
  return this->tie() > other.tie();
}

bool ShardSignatureInstance::operator<=(
    ShardSignatureInstance const &other) const {
  return this->tie() <= other.tie();
}

bool ShardSignatureInstance::operator>=(
    ShardSignatureInstance const &other) const {
  return this->tie() >= other.tie();
}

std::set<OperatorAtomicTaskShardBinding> const &
    ShardSignatureInstance::get_shard_bindings() const {
  return this->bindings;
}

std::tuple<std::set<OperatorAtomicTaskShardBinding> const &>
    ShardSignatureInstance::tie() const {
  return std::tie(this->bindings);
}

std::set<TensorSlotName> shard_signature_instance_get_slot_name_set(
    ShardSignatureInstance const &shard_signature_instance) {
  return require_all_same1(transform(
      shard_signature_instance.get_shard_bindings(),
      [&](OperatorAtomicTaskShardBinding const &b) -> std::set<TensorSlotName> {
        return keys(b.tensor_coords);
      }));
}

ParallelTensorDimDegrees
    parallel_tensor_space_for_shard_signature_instance_and_slot(
        ShardSignatureInstance const &shard_signature_instance,
        TensorSlotName slot_name) {
  std::set<ParallelTensorSpaceCoordinate> task_space_coords =
      transform(shard_signature_instance.get_shard_bindings(),
                [&](OperatorAtomicTaskShardBinding const &b)
                    -> ParallelTensorSpaceCoordinate {
                  return b.tensor_coords.at(slot_name);
                });

  return assert_unwrap(
      strict_parallel_tensor_dim_degrees_for_coord_set(task_space_coords));
}

nlohmann::json format_as(ShardSignatureInstance const &inst) {
  nlohmann::json j = inst;
  return j;
}

std::ostream &operator<<(std::ostream &s, ShardSignatureInstance const &inst) {
  return (s << fmt::to_string(inst));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::ShardSignatureInstance>::operator()(
    ::FlexFlow::ShardSignatureInstance const &inst) const {
  return get_std_hash(inst.tie());
};

} // namespace std

namespace nlohmann {

::FlexFlow::ShardSignatureInstance
    adl_serializer<::FlexFlow::ShardSignatureInstance>::from_json(
        json const &j) {
  return ::FlexFlow::ShardSignatureInstance{
      j.at("bindings").template get<::FlexFlow::ShardSignatureInstance>(),
  };
}

void adl_serializer<::FlexFlow::ShardSignatureInstance>::to_json(
    json &j, ::FlexFlow::ShardSignatureInstance const &t) {
  j["bindings"] = t.get_shard_bindings();
}

} // namespace nlohmann
