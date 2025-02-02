#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H

#include "utils/graph/digraph/digraph.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &);

/**
 * @brief Returns the set of nodes in the graph with no incoming edges.
 */
std::unordered_set<Node> get_initial_nodes(DiGraphView const &graph);

/**
 * @brief Returns the set of nodes in the graph with no outgoing edges.
 */
std::unordered_set<Node> get_terminal_nodes(DiGraphView const &graph);

} // namespace FlexFlow

#endif
