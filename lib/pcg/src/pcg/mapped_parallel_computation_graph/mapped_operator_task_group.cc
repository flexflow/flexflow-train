#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/tuple.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/set_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/contains.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/map_values.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"

namespace FlexFlow {

MappedOperatorTaskGroup::MappedOperatorTaskGroup(
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
        &shard_bindings)
    : shard_bindings(shard_bindings) {
  std::vector<std::unordered_set<TensorSlotName>> binding_slot_sets =
      transform(vector_of(shard_bindings.right_values()),
                [&](OperatorAtomicTaskShardBinding const &s)
                    -> std::unordered_set<TensorSlotName> {
                  return unordered_keys(s.tensor_coords);
                });

  std::unordered_set<TensorSlotName> slot_names =
      require_all_same(binding_slot_sets).value();

  for (TensorSlotName const &slot_name : slot_names) {
    std::vector<OperatorAtomicTaskShardBinding> signatures_for_key =
        vector_of(shard_bindings.right_values());

    std::vector<ParallelTensorSpaceCoordinate> coords_for_key = transform(
        signatures_for_key,
        [&](OperatorAtomicTaskShardBinding const &signature) {
          return ptensor_space_coord_for_slot_name(signature, slot_name);
        });

    std::vector<num_ptensor_parallel_dims_t> coord_dims_for_key =
        transform(coords_for_key, [](ParallelTensorSpaceCoordinate const &c) {
          return ptensor_coord_num_dims(c);
        });

    require_all_same(coord_dims_for_key);
  }
}

bool MappedOperatorTaskGroup::operator==(
    MappedOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedOperatorTaskGroup::operator!=(
    MappedOperatorTaskGroup const &other) const {
  return this->tie() != other.tie();
}

bool MappedOperatorTaskGroup::operator<(
    MappedOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool MappedOperatorTaskGroup::operator>(
    MappedOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool MappedOperatorTaskGroup::operator<=(
    MappedOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool MappedOperatorTaskGroup::operator>=(
    MappedOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::tuple<
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &>
    MappedOperatorTaskGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &
    MappedOperatorTaskGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    get_tensor_bindings_for_slot_name(MappedOperatorTaskGroup const &task_group,
                                      TensorSlotName const &slot_name) {
  std::set<TensorSlotName> slot_names = get_slot_names_for_task_group(task_group);
  ASSERT(contains(slot_names, slot_name));

  std::unordered_map<MachineSpaceCoordinate, ParallelTensorSpaceCoordinate> m =
    map_values(task_group.get_shard_bindings().as_unordered_map(),
               [&](OperatorAtomicTaskShardBinding const &b) -> ParallelTensorSpaceCoordinate {
                 return ptensor_space_coord_for_slot_name(b, slot_name);
               });

  return bidict_from_unstructured_relation(unordered_set_of(m)).reversed();
}

std::set<TensorSlotName> get_slot_names_for_task_group(MappedOperatorTaskGroup const &g) {
  return require_all_same1(
    transform(vector_of(right_entries(g.get_shard_bindings())),
              [&](OperatorAtomicTaskShardBinding const &shard_bindings) -> std::set<TensorSlotName> {
                return keys(shard_bindings.tensor_coords);
              }));
}

nlohmann::json
    mapped_operator_task_group_as_dot_json(MappedOperatorTaskGroup const &m) {

  std::vector<MachineSpaceCoordinate> coordinates =
      sorted(m.get_shard_bindings().left_values());

  return nlohmann::json{
      transform(coordinates,
                [&](MachineSpaceCoordinate const &c) -> std::string {
                  return fmt::format("({}, {})", c.node_idx, c.device_idx);
                }),
  };
}

std::string format_as(::FlexFlow::MappedOperatorTaskGroup const &m) {
  return fmt::format("<MappedOperatorTaskGroup shard_bindings={}>",
                     m.get_shard_bindings());
}

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::MappedOperatorTaskGroup const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedOperatorTaskGroup>::operator()(
    ::FlexFlow::MappedOperatorTaskGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::MappedOperatorTaskGroup
    adl_serializer<::FlexFlow::MappedOperatorTaskGroup>::from_json(
        json const &j) {
  return ::FlexFlow::MappedOperatorTaskGroup{j.template get<
      ::FlexFlow::bidict<::FlexFlow::MachineSpaceCoordinate,
                         ::FlexFlow::OperatorAtomicTaskShardBinding>>()};
}

void adl_serializer<::FlexFlow::MappedOperatorTaskGroup>::to_json(
    json &j, ::FlexFlow::MappedOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
