#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_DATAFLOW_GRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_DATAFLOW_GRAPH_AS_DOT_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_as_dot.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel>
std::string labelled_open_dataflow_graph_as_dot(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &g,
    std::function<std::string(NodeLabel const &)> const &render_node_label,
    std::function<std::string(ValueLabel const &)> const &render_value_label,
    std::function<std::string(DataflowGraphInput const &)> const
        &render_dataflow_graph_input,
    std::function<std::string(DataflowInput const &)> const
        &render_dataflow_input,
    std::function<std::string(DataflowOutput const &)> const
        &render_dataflow_output) {
  std::function<std::string(Node const &)> render_node =
      [&](Node const &n) -> std::string { return render_node_label(g.at(n)); };

  std::function<std::string(DataflowGraphInput const &)>
      render_unlabelled_dataflow_graph_input =
          [&](DataflowGraphInput const &i) {
            return render_value_label(g.at(OpenDataflowValue{i}));
          };

  return open_dataflow_graph_as_dot(static_cast<OpenDataflowGraphView>(g),
                                    render_node,
                                    render_unlabelled_dataflow_graph_input,
                                    render_dataflow_input,
                                    render_dataflow_output);
}

} // namespace FlexFlow

#endif
