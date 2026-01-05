#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H

#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

DynamicOpenDataflowGraph make_empty_dynamic_open_dataflow_graph();

nonnegative_int dynamic_graph_num_nodes(DynamicOpenDataflowGraph const &);

bool full_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &);

bool no_part_of_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &);

std::unordered_multiset<DynamicNodeAttrs> get_dynamic_nodes(DynamicOpenDataflowGraph const &);
std::unordered_multiset<DynamicValueAttrs> get_dynamic_values(DynamicOpenDataflowGraph const &);
std::unordered_set<DynamicNodeInvocation> get_dynamic_invocation_set(DynamicOpenDataflowGraph const &);

DynamicOpenDataflowGraph
  transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const &);

DynamicOpenDataflowGraph
  flatmap_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<std::unordered_set<DynamicNodeInvocation>(DynamicNodeInvocation const &)> const &);

DynamicOpenDataflowGraph 
  dynamic_open_dataflow_graph_from_invocation_set(std::unordered_set<DynamicNodeInvocation> const &);

bool dynamic_open_dataflow_graphs_are_isomorphic(DynamicOpenDataflowGraph const &,
                                                 DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
