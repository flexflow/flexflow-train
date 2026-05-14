#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_OPEN_DATAFLOW_GRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_OPEN_DATAFLOW_GRAPH_AS_DOT_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::string open_dataflow_graph_as_dot(OpenDataflowGraphView const &);
std::string open_dataflow_graph_as_dot(
    OpenDataflowGraphView const &,
    std::function<std::string(Node const &)> const &render_node,
    std::function<std::string(DataflowGraphInput const &)> const
        &render_dataflow_graph_input,
    std::function<std::string(DataflowInput const &)> const
        &render_dataflow_input,
    std::function<std::string(DataflowOutput const &)> const
        &render_dataflow_output);

} // namespace FlexFlow

#endif
