#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SHARD_EXPANSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SHARD_EXPANSION_H

#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation_sharding_info.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs_sharding_info.dtg.h"

namespace FlexFlow {

void require_node_is_shard_expanded(DynamicNodeAttrs const &);
void require_value_is_shard_expanded(DynamicValueAttrs const &);
void require_invocation_is_fully_shard_expanded(DynamicNodeInvocation const &);
void require_graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &);

void require_node_is_ready_for_shard_expansion(DynamicNodeAttrs const &);
void require_value_is_ready_for_shard_expansion(DynamicValueAttrs const &);
void require_invocation_is_ready_for_shard_expansion(
    DynamicNodeInvocation const &);
void require_graph_is_ready_for_shard_expansion(
    DynamicOpenDataflowGraph const &);

[[nodiscard]] DynamicNodeAttrs
    apply_dynamic_node_attrs_sharding_info(DynamicNodeAttrs const &,
                                           MachineSpaceCoordinate const &);

[[nodiscard]] DynamicValueAttrs apply_dynamic_value_attrs_sharding_info(
    DynamicValueAttrs const &, DynamicValueAttrsShardingInfo const &);

[[nodiscard]] DynamicNodeInvocation apply_dynamic_node_invocation_sharding_info(
    DynamicNodeInvocation const &, DynamicNodeInvocationShardingInfo const &);

[[nodiscard]] std::set<DynamicNodeInvocationShardingInfo>
    generate_shard_expansion_for_invocation(DynamicNodeInvocation const &);

[[nodiscard]] std::set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &);

[[nodiscard]] DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
