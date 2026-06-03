#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_DIGRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_DIGRAPH_AS_DOT_H

#include "utils/graph/digraph/digraph_view.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

std::string digraph_as_dot(
    DiGraphView const &,
    std::function<nlohmann::json(Node const &)> const &get_node_label);

} // namespace FlexFlow

#endif
