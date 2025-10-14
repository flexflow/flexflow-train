#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_MAPPED_RUNTIME_TASK_GROUP_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_MAPPED_RUNTIME_TASK_GROUP_H

#include "compiler/mapped_operator_task_group.h"
#include "local-pcg-execution/runtime_atomic_task_shard_binding.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "task-spec/fwb_op_task_type.dtg.h"
#include "task-spec/symbolic_layer_training_tensor_group_signature.dtg.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

struct MappedRuntimeTaskGroup {
  MappedRuntimeTaskGroup() = delete;

  explicit MappedRuntimeTaskGroup(bidict<MachineSpaceCoordinate, RuntimeAtomicTaskShardBinding> const &shard_bindings);

  [[nodiscard]] bool operator==(MappedRuntimeTaskGroup const &) const;
  [[nodiscard]] bool operator!=(MappedRuntimeTaskGroup const &) const;

  [[nodiscard]] bidict<MachineSpaceCoordinate, RuntimeAtomicTaskShardBinding> const &get_shard_bindings() const;

private:
  bidict<MachineSpaceCoordinate, RuntimeAtomicTaskShardBinding> shard_bindings;

private:
  [[nodiscard]] std::tuple<
    decltype(shard_bindings) const &
  > tie() const;

  friend struct ::std::hash<MappedRuntimeTaskGroup>;
};

std::string format_as(::FlexFlow::MappedRuntimeTaskGroup const &);
std::ostream &operator<<(std::ostream &, ::FlexFlow::MappedRuntimeTaskGroup const &);

MappedRuntimeTaskGroup
  lower_mapped_operator_task_group_to_mapped_runtime_task_group(MappedOperatorTaskGroup const &,
                                                                SymbolicLayerTrainingTensorGroupSignature const &,
                                                                FwbOpTaskType);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MappedRuntimeTaskGroup> {
  size_t operator()(::FlexFlow::MappedRuntimeTaskGroup const &) const;
};

} // namespace std

#endif
