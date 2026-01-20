#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_ANCESTORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_ANCESTORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {
/**
 * @brief Computes the set of all ancestors of a given node `n` in a directed
 *graph, which is the set of all nodes `m` for which a directed path from `n` to
 *`m` exists.
 *
 * @note `n` is not considered to be its own ancestor, and is thus not
 *included in the returned set.
 **/
std::unordered_set<Node> get_ancestors(DiGraphView const &g, Node const &n);

} // namespace FlexFlow

#endif
