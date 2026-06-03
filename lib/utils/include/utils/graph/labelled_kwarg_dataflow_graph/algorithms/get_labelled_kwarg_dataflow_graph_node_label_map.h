#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_NODE_LABEL_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_NODE_LABEL_MAP_H

#include "utils/containers/generate_map.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
std::unordered_map<Node, NodeLabel>
    get_labelled_kwarg_dataflow_graph_node_label_map(
        LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const
            &g) {
  return generate_map(get_nodes(g),
                      [&](Node const &n) -> NodeLabel { return g.at(n); });
}

} // namespace FlexFlow

#endif
