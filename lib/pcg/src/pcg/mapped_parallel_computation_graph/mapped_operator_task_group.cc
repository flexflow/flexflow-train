#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_atomic_task_shard_binding.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.h"
#include "pcg/mapped_parallel_computation_graph/parallelism_operator_atomic_task_shard_binding.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "utils/bidict/algorithms/bidict_right_entries.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"
#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/binary_relation/binary_relation_is_biunique.h"
#include "utils/containers/all_of.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/containers/contains.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/transform_pairs.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/tuple.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

MappedOperatorTaskGroup::MappedOperatorTaskGroup(
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
        &shard_bindings)
    : shard_bindings(shard_bindings) {

  std::set<TensorSlotName> slot_names = [&]() -> std::set<TensorSlotName> {
    std::vector<std::set<TensorSlotName>> binding_slot_sets = transform(
        vector_of(shard_bindings.right_values()),
        [&](OperatorAtomicTaskShardBinding const &s)
            -> std::set<TensorSlotName> { return keys(s.tensor_coords); });

    return require_all_same1(binding_slot_sets);
  }();

  std::set<MappedOperatorAtomicTaskShardBinding> mapped_bindings =
      transform_pairs(unstructured_relation_from_bidict(shard_bindings),
                      [&](MachineSpaceCoordinate const &mc,
                          OperatorAtomicTaskShardBinding const &b)
                          -> MappedOperatorAtomicTaskShardBinding {
                        return MappedOperatorAtomicTaskShardBinding{
                            /*tensor_coords=*/b.tensor_coords,
                            /*machine_coord=*/mc,
                        };
                      });

  bool is_valid_standard_op_task_group =
      all_of(slot_names, [&](TensorSlotName slot_name) -> bool {
        return mapped_op_task_shard_bindings_are_unique_on_slot(mapped_bindings,
                                                                slot_name);
      });

  std::set<TensorSlotName> valid_parallelism_op_slots = {
      TensorSlotName::INPUT, TensorSlotName::OUTPUT};

  bool is_valid_expansive_parallelism_op_task_group =
      slot_names == valid_parallelism_op_slots &&
      mapped_op_task_shard_bindings_are_unique_on_slot(mapped_bindings,
                                                       TensorSlotName::INPUT) &&
      mapped_op_task_shard_bindings_are_strictly_k_unique_on_slot(
          mapped_bindings, TensorSlotName::OUTPUT);

  bool is_valid_contractive_parallelism_op_task_group =
      slot_names == valid_parallelism_op_slots &&
      mapped_op_task_shard_bindings_are_unique_on_slot(
          mapped_bindings, TensorSlotName::OUTPUT) &&
      mapped_op_task_shard_bindings_are_strictly_k_unique_on_slot(
          mapped_bindings, TensorSlotName::INPUT);

  ASSERT(is_valid_standard_op_task_group ||
         is_valid_expansive_parallelism_op_task_group ||
         is_valid_contractive_parallelism_op_task_group);

  for (TensorSlotName slot_name : slot_names) {
    std::vector<ParallelTensorSpaceCoordinate> coords_for_key = transform(
        vector_of(shard_bindings.right_values()),
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

std::set<MappedOperatorAtomicTaskShardBinding>
    get_mapped_operator_atomic_task_shard_bindings_for_task_group(
        MappedOperatorTaskGroup const &g) {
  return transform_pairs(
      unstructured_relation_from_bidict(g.get_shard_bindings()),
      [&](MachineSpaceCoordinate const &mc,
          OperatorAtomicTaskShardBinding const &b)
          -> MappedOperatorAtomicTaskShardBinding {
        return MappedOperatorAtomicTaskShardBinding{
            /*tensor_coords=*/b.tensor_coords,
            /*machine_coord=*/mc,
        };
      });
}

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    get_tensor_bindings_for_slot_name(MappedOperatorTaskGroup const &task_group,
                                      TensorSlotName const &slot_name) {
  std::set<TensorSlotName> slot_names =
      get_slot_names_for_task_group(task_group);
  ASSERT(contains(slot_names, slot_name));

  std::map<MachineSpaceCoordinate, ParallelTensorSpaceCoordinate> m =
      map_values(task_group.get_shard_bindings().as_map(),
                 [&](OperatorAtomicTaskShardBinding const &b)
                     -> ParallelTensorSpaceCoordinate {
                   return ptensor_space_coord_for_slot_name(b, slot_name);
                 });

  return bidict_from_unstructured_relation(set_of(m)).reversed();
}

std::set<TensorSlotName>
    get_slot_names_for_task_group(MappedOperatorTaskGroup const &g) {
  return require_all_same1(
      transform(vector_of(bidict_right_entries(g.get_shard_bindings())),
                [&](OperatorAtomicTaskShardBinding const &shard_bindings)
                    -> std::set<TensorSlotName> {
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

nlohmann::json format_as(::FlexFlow::MappedOperatorTaskGroup const &m) {
  return m;
}

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::MappedOperatorTaskGroup const &x) {
  return (s << fmt::to_string(x));
}

MappedOperatorTaskGroup mapped_op_task_group_from_standard_op_group(
    MappedStandardOperatorTaskGroup const &g) {
  return MappedOperatorTaskGroup{
      g.get_shard_bindings(),
  };
}

MappedOperatorTaskGroup mapped_op_task_group_from_parallelism_op_group(
    MappedParallelismOperatorTaskGroup const &g) {
  return MappedOperatorTaskGroup{
      bidict_transform_values(
          g.get_shard_bindings(),
          [](ParallelismOperatorAtomicTaskShardBinding const &b)
              -> OperatorAtomicTaskShardBinding {
            return operator_atomic_task_shard_binding_from_parallelism_op_binding(
                b);
          }),
  };
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
  return ::FlexFlow::MappedOperatorTaskGroup{
      j.template get<
          ::FlexFlow::bidict<::FlexFlow::MachineSpaceCoordinate,
                             ::FlexFlow::OperatorAtomicTaskShardBinding>>(),
  };
}

void adl_serializer<::FlexFlow::MappedOperatorTaskGroup>::to_json(
    json &j, ::FlexFlow::MappedOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
