#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_SUBGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_SUBGRAPH_H

#include "utils/containers/contains.h"
#include "utils/containers/filter_keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_subgraph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_node_label_map.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_output_label_map.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>
    get_labelled_kwarg_dataflow_graph_subgraph(
        LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const
            &g,
        std::unordered_set<Node> const &subgraph_nodes) {
  KwargDataflowGraphView<SlotName> unlabelled_subgraph =
      get_kwarg_dataflow_graph_subgraph(g, subgraph_nodes);

  std::unordered_map<Node, NodeLabel> g_node_labelling =
      get_labelled_kwarg_dataflow_graph_node_label_map(g);

  std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel>
      g_output_labelling =
          get_labelled_kwarg_dataflow_graph_output_label_map(g);

  return kwarg_dataflow_graph_view_with_labelling(
      unlabelled_subgraph,
      restrict_keys(g_node_labelling, subgraph_nodes),
      filter_keys(g_output_labelling,
                  [&](KwargDataflowOutput<SlotName> const &o) -> bool {
                    return contains(subgraph_nodes, o.node);
                  }));
}

} // namespace FlexFlow

#endif
