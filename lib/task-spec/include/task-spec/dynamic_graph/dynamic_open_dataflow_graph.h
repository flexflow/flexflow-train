#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_OPEN_DATAFLOW_GRAPH_H

#include "task-spec/dynamic_graph/dynamic_graph_edge.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"

namespace FlexFlow {

DynamicOpenDataflowGraph make_empty_dynamic_open_dataflow_graph();

void check_dynamic_open_dataflow_graph_is_valid(
    DynamicOpenDataflowGraph const &);

nonnegative_int dynamic_graph_num_nodes(DynamicOpenDataflowGraph const &);

bool full_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &,
    std::function<bool(DynamicNodeAttrs const &)> const &,
    std::function<bool(DynamicValueAttrs const &)> const &,
    std::function<bool(DynamicTensorSlot const &)> const &);

bool no_part_of_dynamic_graph_satisfies(
    DynamicOpenDataflowGraph const &,
    std::function<bool(DynamicNodeAttrs const &)> const &,
    std::function<bool(DynamicValueAttrs const &)> const &,
    std::function<bool(DynamicTensorSlot const &)> const &);

std::unordered_multiset<DynamicNodeAttrs>
    get_dynamic_nodes(DynamicOpenDataflowGraph const &);
std::unordered_multiset<DynamicValueAttrs>
    get_dynamic_values(DynamicOpenDataflowGraph const &);
std::unordered_multiset<DynamicTensorSlot>
    get_dynamic_tensor_slots(DynamicOpenDataflowGraph const &);
std::unordered_set<DynamicNodeInvocation>
    get_dynamic_invocation_set(DynamicOpenDataflowGraph const &);

std::unordered_set<DynamicGraphEdge>
    get_dynamic_graph_edges(DynamicOpenDataflowGraph const &);
std::unordered_set<DynamicGraphEdge>
    get_dynamic_graph_edges_incoming_to_invocation(
        DynamicOpenDataflowGraph const &, DynamicNodeInvocation const &);
std::unordered_set<DynamicGraphEdge>
    get_dynamic_graph_edges_outgoing_from_invocation(
        DynamicOpenDataflowGraph const &, DynamicNodeInvocation const &);

std::unordered_set<InternalDynamicSlotSite>
    get_internal_dynamic_slot_sites(DynamicOpenDataflowGraph const &);

std::unordered_set<DynamicSlotSite>
    get_dynamic_slot_sites(DynamicOpenDataflowGraph const &);

DynamicSlotSite
    dynamic_graph_find_source_of_value(DynamicOpenDataflowGraph const &,
                                       DynamicValueAttrs const &);
std::unordered_set<InternalDynamicSlotSite>
    dynamic_graph_find_sinks_of_value(DynamicOpenDataflowGraph const &,
                                      DynamicValueAttrs const &);

std::optional<DynamicValueAttrs>
    find_output_value_attrs(DynamicOpenDataflowGraph const &,
                            dynamic_tensor_guid_t,
                            std::optional<DynamicTensorRole> const &);

DynamicOpenDataflowGraph transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const
        &);

DynamicOpenDataflowGraph flatmap_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<std::unordered_set<DynamicNodeInvocation>(
        DynamicNodeInvocation const &)> const &);

DynamicOpenDataflowGraph dynamic_open_dataflow_graph_from_invocation_set(
    std::unordered_set<DynamicNodeInvocation> const &);

std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                         DynamicValueAttrs,
                                         int,
                                         DynamicTensorSlot>,
          bidict<Node, DynamicNodeInvocation>>
    labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
        DynamicOpenDataflowGraph const &);

bool dynamic_open_dataflow_graphs_are_isomorphic(
    DynamicOpenDataflowGraph const &, DynamicOpenDataflowGraph const &);

std::string
    dynamic_open_dataflow_graph_as_dot(DynamicOpenDataflowGraph const &);
void debug_print_dynamic_open_dataflow_graph_as_dot(
    DynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
