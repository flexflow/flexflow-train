#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &n) {
  return n.device_coord.has_value(); 
}

bool value_is_shard_expanded(DynamicValueAttrs const &n) {
  return n.shard_coord.has_value();
}

bool no_part_of_graph_is_shard_expanded(DynamicOpenDataflowGraph const &g) {
  return no_part_of_dynamic_graph_satisfies(
    g, node_is_shard_expanded, value_is_shard_expanded);
}

bool graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &g) {
  return full_dynamic_graph_satisfies(
    g, node_is_shard_expanded, value_is_shard_expanded);
}

std::unordered_set<DynamicNodeInvocation> 
  perform_shard_expansion_for_invocation(DynamicNodeInvocation const &) {
  // TODO(@lockshaw)(#pr): 
  NOT_IMPLEMENTED(); 
}

} // namespace FlexFlow
