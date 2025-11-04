#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H

#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

bool full_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &);

bool no_part_of_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &);

std::unordered_set<DynamicNodeAttrs> get_dynamic_node_set();
std::unordered_set<DynamicNodeInvocation> get_dynamic_invocation_set(DynamicOpenDataflowGraph const &);

DynamicOpenDataflowGraph
  transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const &);

} // namespace FlexFlow

#endif
