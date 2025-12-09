#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISMS_BETWEEN_OPEN_KWARG_DATAFLOW_GRAPHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISMS_BETWEEN_OPEN_KWARG_DATAFLOW_GRAPHS_H

#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_unused_open_kwarg_dataflow_graph_inputs.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::optional<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
    find_isomorphism_under_sink_node_mapping(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &src_g,
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &dst_g,
        bidict<Node, Node> const &sink_node_mapping,
        bidict<DataflowGraphInput, DataflowGraphInput> const
            &unused_graph_inputs_mapping) {
  {
    std::unordered_set<Node> already_mapped_src_nodes =
        left_entries(sink_node_mapping);
    std::unordered_set<Node> src_g_sink_nodes = get_terminal_nodes(src_g);
    ASSERT(already_mapped_src_nodes == src_g_sink_nodes);
  }

  {
    std::unordered_set<Node> already_mapped_dst_nodes =
        right_entries(sink_node_mapping);
    std::unordered_set<Node> dst_g_sink_nodes = get_terminal_nodes(dst_g);
    ASSERT(already_mapped_dst_nodes == dst_g_sink_nodes);
  }

  {
    std::unordered_set<DataflowGraphInput> already_mapped_src_inputs =
        right_entries(unused_graph_inputs_mapping);
    std::unordered_set<DataflowGraphInput> src_g_unused_inputs =
        get_unused_open_kwarg_dataflow_graph_inputs(src_g);
    ASSERT(already_mapped_src_inputs == src_g_unused_inputs);
  }

  {
    std::unordered_set<DataflowGraphInput> already_mapped_dst_inputs =
        right_entries(unused_graph_inputs_mapping);
    std::unordered_set<DataflowGraphInput> dst_g_unused_inputs =
        get_unused_open_kwarg_dataflow_graph_inputs(dst_g);
    ASSERT(already_mapped_dst_inputs == dst_g_unused_inputs);
  }

  std::optional<OpenKwargDataflowGraphIsomorphism<GraphInputName>> result =
      OpenKwargDataflowGraphIsomorphism<GraphInputName>{
          {},
          unused_graph_inputs_mapping,
      };

  auto fail = [&]() -> void { result = std::nullopt; };

  auto has_failed = [&]() -> bool { return result == std::nullopt; };

  std::function<void(Node const &, Node const &)> unify_nodes;
  std::function<void(OpenKwargDataflowEdge<GraphInputName, SlotName> const &, OpenKwargDataflowEdge<GraphInputName, SlotName> const &)>
      unify_edges;
  std::function<void(KwargDataflowGraphInput<GraphInputName> const &, KwargDataflowGraphInput<GraphInputName> const &)>
      unify_graph_inputs;
  std::function<void(OpenKwargDataflowValue<GraphInputName, SlotName> const &, OpenKwargDataflowValue<GraphInputName, SlotName> const &)>
      unify_values;
  std::function<void(KwargDataflowOutput<SlotName> const &, KwargDataflowOutput<SlotName> const &)>
      unify_outputs;

  unify_outputs = [&](KwargDataflowOutput<SlotName> const &src_output,
                      KwargDataflowOutput<SlotName> const &dst_output) {
    if (has_failed()) {
      return;
    }

    if (src_output.idx != dst_output.idx) {
      result = std::nullopt;
      return;
    }

    unify_nodes(src_output.node, dst_output.node);
  };

  unify_values = [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &src_val,
                     OpenKwargDataflowValue<GraphInputName, SlotName> const &dst_val) {
    if (has_failed()) {
      return;
    }

    if (src_val.index() != dst_val.index()) {
      fail();
      return;
    }

    if (src_val.is_internal()) {
      unify_outputs(src_val.require_internal(),
                    dst_val.require_internal());
    } else {
      unify_graph_inputs(src_val.require_external(),
                         dst_val.require_external());
    }
  };

  unify_graph_inputs = [&](KwargDataflowGraphInput<GraphInputName> const &src,
                           KwargDataflowGraphInput<GraphInputName> const &dst) {
    if (has_failed()) {
      return;
    }

    if (result->input_mapping.contains_l(src) &&
        result->input_mapping.at_l(src) != dst) {
      fail();
      return;
    }
    if (result->input_mapping.contains_r(dst) &&
        result->input_mapping.at_r(dst) != src) {
      fail();
      return;
    }

    result->input_mapping.equate(src, dst);
  };

  unify_edges = [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &src_edge,
                    OpenKwargDataflowEdge<GraphInputName, SlotName> const &dst_edge) {
    if (has_failed()) {
      return;
    }

    ASSERT(get_open_kwarg_dataflow_edge_dst(src_edge).idx ==
           get_open_kwarg_dataflow_edge_dst(dst_edge).idx);
    ASSERT(
        get_open_kwarg_dataflow_edge_dst(src_edge).node ==
        result->node_mapping.at_r(get_open_kwarg_dataflow_edge_dst(dst_edge).node));

    unify_values(get_open_dataflow_edge_src(src_edge),
                 get_open_dataflow_edge_src(dst_edge));
  };

  unify_nodes = [&](Node const &src_node, Node const &dst_node) {
    if (has_failed()) {
      return;
    }

    if (result->node_mapping.contains(src_node, dst_node)) {
      return;
    }

    if (result->node_mapping.contains_l(src_node) &&
        result->node_mapping.at_l(src_node) != dst_node) {
      fail();
      return;
    }
    if (result->node_mapping.contains_r(dst_node) &&
        result->node_mapping.at_r(dst_node) != src_node) {
      fail();
      return;
    }

    result->node_mapping.equate(src_node, dst_node);

    std::vector<OpenKwargDataflowEdge<GraphInputName, SlotName>> src_incoming_edges =
        get_incoming_edges(src_g, src_node);
    std::vector<OpenKwargDataflowEdge<GraphInputName, SlotName>> dst_incoming_edges =
        get_incoming_edges(dst_g, dst_node);

    if (src_incoming_edges.size() != dst_incoming_edges.size()) {
      fail();
      return;
    }

    for (auto const &[src_edge, dst_edge] :
         zip(src_incoming_edges, dst_incoming_edges)) {
      unify_edges(src_edge, dst_edge);
    }
  };

  for (auto const &[src_node, dst_node] : sink_node_mapping) {
    unify_nodes(src_node, dst_node);
  }

  return result;
}


template <typename GraphInputName, typename SlotName>
std::unordered_set<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
    find_isomorphisms_between_open_kwarg_dataflow_graphs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &src,
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &dst)
{
  std::unordered_set<OpenKwargDataflowGraphIsomorphism<GraphInputName>> result;

  std::vector<Node> src_sink_nodes = vector_of(get_terminal_nodes(src));
  std::unordered_set<Node> dst_sink_nodes = get_terminal_nodes(dst);

  if (src_sink_nodes.size() != dst_sink_nodes.size()) {
    return {};
  }

  std::vector<KwargDataflowGraphInput<GraphInputName>> src_unused_graph_inputs =
      vector_of(get_unused_open_kwarg_dataflow_graph_inputs(src));
  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> dst_unused_graph_inputs =
      get_unused_open_kwarg_dataflow_graph_inputs(dst);

  if (src_unused_graph_inputs.size() != dst_unused_graph_inputs.size()) {
    return {};
  }

  for (std::vector<Node> const &dst_sink_nodes :
       get_all_permutations(dst_sink_nodes)) {

    bidict<Node, Node> sink_node_mapping =
        bidict_from_keys_and_values(src_sink_nodes, dst_sink_nodes);

    for (std::vector<KwargDataflowGraphInput<GraphInputName>> const &dst_unused_graph_inputs :
         get_all_permutations(dst_unused_graph_inputs)) {

      bidict<KwargDataflowGraphInput<GraphInputName>, KwargDataflowGraphInput<GraphInputName>>
          unused_graph_inputs_mapping = bidict_from_keys_and_values(
              src_unused_graph_inputs, dst_unused_graph_inputs);

      std::optional<OpenKwargDataflowGraphIsomorphism<GraphInputName>> found =
          find_isomorphism_under_sink_node_mapping(
              src, dst, sink_node_mapping, unused_graph_inputs_mapping);

      if (found.has_value()) {
        ASSERT(is_isomorphic_under(src, dst, found.value()));

        result.insert(found.value());
      }
    }
  }

  return result;
}


} // namespace FlexFlow

#endif
