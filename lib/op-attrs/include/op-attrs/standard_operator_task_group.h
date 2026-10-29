#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_STANDARD_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_STANDARD_OPERATOR_TASK_GROUP_H

#include "op-attrs/abstracted_operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

struct StandardOperatorTaskGroup {
  StandardOperatorTaskGroup() = delete;

  explicit StandardOperatorTaskGroup(
      std::set<AbstractedOperatorAtomicTaskShardBinding> const &shard_bindings);

  [[nodiscard]] bool operator==(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(StandardOperatorTaskGroup const &) const;

  [[nodiscard]] bool operator<(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator<=(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>=(StandardOperatorTaskGroup const &) const;

  [[nodiscard]] std::set<AbstractedOperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  std::set<AbstractedOperatorAtomicTaskShardBinding> bindings;

private:
  [[nodiscard]] std::tuple<decltype(bindings) const &> tie() const;

  friend struct ::std::hash<StandardOperatorTaskGroup>;
};

OperatorTaskSpace task_space_for_standard_operator_task_group(
    StandardOperatorTaskGroup const &);

ParallelTensorDimDegrees
    parallel_tensor_space_for_standard_operator_task_group_and_slot(
        StandardOperatorTaskGroup const &, TensorSlotName slot_name);

ShardSignatureInstance
    shard_signature_instance_from_standard_operator_task_group(
        StandardOperatorTaskGroup const &);

OperatorSpaceToParallelTensorSpaceMapping
    standard_operator_task_group_get_operator_to_ptensor_mapping(
        StandardOperatorTaskGroup const &, TensorSlotName slot_name);

ParallelTensorSpaceToParallelTensorSpaceMapping
    standard_operator_task_group_get_ptensor_to_ptensor_mapping(
        StandardOperatorTaskGroup const &,
        TensorSlotName lhs_slot_name,
        TensorSlotName rhs_slot_name);

StandardOperatorTaskGroup restrict_standard_operator_task_group_to_slots(
    StandardOperatorTaskGroup const &, std::set<TensorSlotName> const &);

StandardOperatorTaskGroup
    standard_operator_task_group_from_shard_signature_instance(
        ShardSignatureInstance const &,
        std::function<TaskSpaceCoordinate(
            OperatorAtomicTaskShardBinding const &)> const &);

OperatorAtomicTaskShardBinding
    standard_op_task_group_get_binding_for_task_space_coord(
        StandardOperatorTaskGroup const &, TaskSpaceCoordinate const &);

nlohmann::json format_as(::FlexFlow::StandardOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::StandardOperatorTaskGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::StandardOperatorTaskGroup> {
  size_t operator()(::FlexFlow::StandardOperatorTaskGroup const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::StandardOperatorTaskGroup> {
  static ::FlexFlow::StandardOperatorTaskGroup from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::StandardOperatorTaskGroup const &t);
};

} // namespace nlohmann

#endif
