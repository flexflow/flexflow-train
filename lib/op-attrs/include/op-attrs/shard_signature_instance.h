#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHARD_SIGNATURE_INSTANCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHARD_SIGNATURE_INSTANCE_H

#include "op-attrs/abstracted_operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "utils/bidict/bidict.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

struct ShardSignatureInstance {
  ShardSignatureInstance() = delete;

  explicit ShardSignatureInstance(
      std::set<OperatorAtomicTaskShardBinding> const &shard_bindings);

  [[nodiscard]] bool operator==(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator!=(ShardSignatureInstance const &) const;

  [[nodiscard]] bool operator<(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator>(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator<=(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator>=(ShardSignatureInstance const &) const;

  [[nodiscard]] std::set<OperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  std::set<OperatorAtomicTaskShardBinding> bindings;

private:
  [[nodiscard]] std::tuple<decltype(bindings) const &> tie() const;

  friend struct ::std::hash<ShardSignatureInstance>;
};

std::set<TensorSlotName>
    shard_signature_instance_get_slot_name_set(ShardSignatureInstance const &);

ParallelTensorDimDegrees
    parallel_tensor_space_for_shard_signature_instance_and_slot(
        ShardSignatureInstance const &, TensorSlotName slot_name);

nlohmann::json format_as(ShardSignatureInstance const &);
std::ostream &operator<<(std::ostream &, ShardSignatureInstance const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::ShardSignatureInstance> {
  size_t operator()(::FlexFlow::ShardSignatureInstance const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::ShardSignatureInstance> {
  static ::FlexFlow::ShardSignatureInstance from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::ShardSignatureInstance const &t);
};

} // namespace nlohmann

#endif
