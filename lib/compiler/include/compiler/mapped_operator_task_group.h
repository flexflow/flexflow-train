#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MAPPED_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MAPPED_OPERATOR_TASK_GROUP_H

#include "compiler/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "utils/bidict/bidict.h"
#include "compiler/mapped_task_signature_tensor_key.dtg.h"

namespace FlexFlow {

struct MappedOperatorTaskGroup {
  MappedOperatorTaskGroup() = delete;

  explicit MappedOperatorTaskGroup(bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &shard_bindings);

  [[nodiscard]] bool operator==(MappedOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(MappedOperatorTaskGroup const &) const;

  [[nodiscard]] bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &get_shard_bindings() const;

private:
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> shard_bindings;

private:
  [[nodiscard]] std::tuple<
    decltype(shard_bindings) const &
  > tie() const;

  friend struct ::std::hash<MappedOperatorTaskGroup>;
};

std::string format_as(::FlexFlow::MappedOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &, ::FlexFlow::MappedOperatorTaskGroup const &);

MappedOperatorTaskGroup
  mapped_operator_task_group_from_machine_view(
    ComputationGraphOpAttrs const &,
    std::vector<ParallelTensorDimDegrees> const &,
    MachineView const &);


} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MappedOperatorTaskGroup> {
  size_t operator()(::FlexFlow::MappedOperatorTaskGroup const &) const;
};

} // namespace std
#endif
