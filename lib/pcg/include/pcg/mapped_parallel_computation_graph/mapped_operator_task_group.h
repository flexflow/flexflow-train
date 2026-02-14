#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_OPERATOR_TASK_GROUP_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "utils/bidict/bidict.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

struct MappedOperatorTaskGroup {
  MappedOperatorTaskGroup() = delete;

  explicit MappedOperatorTaskGroup(
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
          &shard_bindings);

  [[nodiscard]] bool operator==(MappedOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(MappedOperatorTaskGroup const &) const;

  [[nodiscard]] bidict<MachineSpaceCoordinate,
                       OperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> shard_bindings;

private:
  [[nodiscard]] std::tuple<decltype(shard_bindings) const &> tie() const;

  friend struct ::std::hash<MappedOperatorTaskGroup>;
};

std::string format_as(::FlexFlow::MappedOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::MappedOperatorTaskGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MappedOperatorTaskGroup> {
  size_t operator()(::FlexFlow::MappedOperatorTaskGroup const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::MappedOperatorTaskGroup> {
  static ::FlexFlow::MappedOperatorTaskGroup from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::MappedOperatorTaskGroup const &t);
};

} // namespace nlohmann

#endif
