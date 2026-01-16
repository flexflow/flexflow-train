#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_INITIAL_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_INITIAL_NODES_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

/**
 * @brief Returns the set of nodes in the graph with no incoming edges.
 */
std::unordered_set<Node> get_initial_nodes(DiGraphView const &graph);

} // namespace FlexFlow

#endif
