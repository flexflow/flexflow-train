#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SHARD_EXPANSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SHARD_EXPANSION_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs_sharding_info.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation_sharding_info.dtg.h"

namespace FlexFlow {

[[nodiscard]] bool node_is_shard_expanded(DynamicNodeAttrs const &);
[[nodiscard]] bool value_is_shard_expanded(DynamicValueAttrs const &);
[[nodiscard]] bool invocation_is_fully_shard_expanded(DynamicNodeInvocation const &);

[[nodiscard]] bool node_is_ready_for_shard_expansion(DynamicNodeAttrs const &);
[[nodiscard]] bool value_is_ready_for_shard_expansion(DynamicValueAttrs const &);
[[nodiscard]] bool invocation_is_ready_for_shard_expansion(DynamicNodeInvocation const &);

[[nodiscard]] bool no_part_of_graph_is_shard_expanded(DynamicOpenDataflowGraph const &);
[[nodiscard]] bool graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &);
[[nodiscard]] bool graph_is_ready_for_shard_expansion(DynamicOpenDataflowGraph const &);

[[nodiscard]] DynamicNodeAttrs apply_dynamic_node_attrs_sharding_info(
  DynamicNodeAttrs const &,
  MachineSpaceCoordinate const &);

[[nodiscard]] DynamicValueAttrs apply_dynamic_value_attrs_sharding_info(
  DynamicValueAttrs const &,
  DynamicValueAttrsShardingInfo const &);

[[nodiscard]] DynamicNodeInvocation apply_dynamic_node_invocation_sharding_info(
  DynamicNodeInvocation const &,
  DynamicNodeInvocationShardingInfo const &);

[[nodiscard]] std::unordered_set<DynamicNodeInvocationShardingInfo>
  generate_shard_expansion_for_invocation(DynamicNodeInvocation const &);

[[nodiscard]] std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &);

[[nodiscard]] DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
