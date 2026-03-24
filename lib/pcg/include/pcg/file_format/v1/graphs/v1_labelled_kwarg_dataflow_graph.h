#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_GRAPHS_V1_LABELLED_KWARG_DATAFLOW_GRAPH_H

#include "pcg/file_format/v1/graphs/v1_kwarg_dataflow_graph.h"
#include "pcg/file_format/v1/graphs/v1_labelled_kwarg_dataflow_graph.dtg.h"
#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_map_from_pairs.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/nonnegative_int/nonnegative_int.h"

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
std::pair<LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>,
          std::unordered_map<nonnegative_int, Node>>
    from_v1(V1LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName> const
                &v1) {
  auto [graph_view, node_map] = from_v1(v1.graph);

  std::unordered_map<Node, NodeLabel> node_labels = map_keys(
      v1.node_labels, [&](nonnegative_int n) { return node_map.at(n); });
  std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel> value_labels;

  return std::pair{kwarg_dataflow_graph_view_with_labelling(
                       graph_view, node_labels, value_labels),
                   node_map};
}

} // namespace FlexFlow

#endif
