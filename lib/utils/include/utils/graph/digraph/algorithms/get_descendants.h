#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DESCENDANTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DESCENDANTS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

/**
 * @brief Computes the set of all descendants of a given node in a directed
 * graph.
 *
 * @note `starting_node` is not considered to be its own descendant, and is thus
 * not included in the returned set.
 **/
std::unordered_set<Node> get_descendants(DiGraphView const &g,
                                         Node const &starting_node);

} // namespace FlexFlow

#endif
