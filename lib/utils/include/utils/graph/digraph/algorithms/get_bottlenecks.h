#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_BOTTLENECKS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_BOTTLENECKS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

/**
 * @brief Returns the bottlenecks of the graph.
 *
 * A bottleneck is a node through which all paths from any sink to any source
 must pass.

 * @note
 * The graph must be acyclic and singly connected.
 * Note that, under the definition of bottleneck, a source / sink is a
 bottleneck if and only if it's the unique source / sink of the graph.
 */
std::unordered_set<Node> get_bottlenecks(DiGraphView const &g);

} // namespace FlexFlow

#endif
