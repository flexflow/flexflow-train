#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_AS_DOT_H

#include "utils/dot/dot_file.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"

namespace FlexFlow {

std::string dataflow_graph_as_dot(
    DataflowGraphView const &,
    std::optional<std::function<nlohmann::json(Node const &)>> const
        &get_node_label = std::nullopt,
    std::optional<std::function<nlohmann::json(DataflowInput const &)>> const
        &get_input_label = std::nullopt,
    std::optional<std::function<nlohmann::json(DataflowOutput const &)>> const
        &get_output_label = std::nullopt);

void dataflow_graph_as_dot(
    DotFile<std::string> &,
    DataflowGraphView const &,
    std::optional<std::function<nlohmann::json(Node const &)>> const
        &get_node_label = std::nullopt,
    std::optional<std::function<nlohmann::json(DataflowInput const &)>> const
        &get_input_label = std::nullopt,
    std::optional<std::function<nlohmann::json(DataflowOutput const &)>> const
        &get_output_label = std::nullopt);

} // namespace FlexFlow

#endif
