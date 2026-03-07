#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_KWARG_DATAFLOW_GRAPH_H

#include "pcg/file_format/v1/graphs/v1_kwarg_dataflow_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_kwarg_dataflow_graph.dtg.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
std::pair<V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>,
          bidict<nonnegative_int, Node>>
    to_v1_including_node_numbering(
        LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const
            &g) {

  bidict<nonnegative_int, Node> nodes = bidict_from_enumerating(get_nodes(g));

  V1KwargDataflowGraph<SlotName> unlabelled = to_v1(g, nodes.reversed());

  std::unordered_map<nonnegative_int, NodeLabel> node_labels = map_values(
      nodes.as_unordered_map(), [&](Node const &n) { return g.at(n); });

  std::unordered_map<nonnegative_int, std::unordered_map<SlotName, OutputLabel>>
      output_labels = map_values(
          nodes.as_unordered_map(),
          [&](Node const &n) -> std::unordered_map<SlotName, OutputLabel> {
            return map_values(
                get_outgoing_kwarg_dataflow_outputs_for_node(g, n),
                [&](KwargDataflowOutput<SlotName> const &o) {
                  return g.at(o);
                });
          });

  return {
      V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>{
          node_labels, output_labels, unlabelled},
      nodes,
  };
}

template <typename NodeLabel, typename OutputLabel, typename SlotName>
V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName> to_v1(
    LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const &g) {
  return to_v1_including_node_numbering(g).first;
}

template <typename NodeLabel, typename OutputLabel, typename SlotName>
LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName> from_v1(
    V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName> const &v1) {
  // Build incoming-edge map
  std::unordered_map<nonnegative_int, std::vector<V1GraphEdge<SlotName>>>
      incoming;
  for (nonnegative_int const &n : v1.graph.nodes) {
    incoming[n] = {};
  }
  for (V1GraphEdge<SlotName> const &e : v1.graph.edges) {
    incoming[e.dstNode].push_back(e);
  }

  // Build a DiGraph with V1 indices as Node raw_uids to get topological order
  DiGraph dg = DiGraph::create<AdjacencyDiGraph>();
  for (nonnegative_int const &n : v1.graph.nodes) {
    dg.add_node_unsafe(Node{static_cast<size_t>(n.unwrap_nonnegative())});
  }
  for (V1GraphEdge<SlotName> const &e : v1.graph.edges) {
    dg.add_edge(DirectedEdge{
        Node{static_cast<size_t>(e.srcNode.unwrap_nonnegative())},
        Node{static_cast<size_t>(e.dstNode.unwrap_nonnegative())}});
  }

  auto g = LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>::
      template create<UnorderedSetLabelledOpenKwargDataflowGraph<NodeLabel,
                                                                 OutputLabel,
                                                                 int,
                                                                 SlotName>>();

  std::unordered_map<nonnegative_int, Node> node_map;
  for (Node const &topo_node : get_topological_ordering(dg)) {
    nonnegative_int v1_idx{static_cast<size_t>(topo_node.raw_uid)};

    std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> inputs;
    for (V1GraphEdge<SlotName> const &e : incoming.at(v1_idx)) {
      inputs.emplace(
          e.dstSlot,
          KwargDataflowOutput<SlotName>{node_map.at(e.srcNode), e.srcSlot});
    }

    KwargNodeAddedResult<SlotName> result = g.add_node(
        v1.node_labels.at(v1_idx), inputs, v1.output_labels.at(v1_idx));

    node_map.emplace(v1_idx, result.node);
  }

  return g;
}

} // namespace FlexFlow

#endif
