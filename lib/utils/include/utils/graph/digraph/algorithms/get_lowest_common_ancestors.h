#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_LOWEST_COMMON_ANCESTORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_LOWEST_COMMON_ANCESTORS_H

#include "utils/graph/digraph/digraph_view.h"
#include <optional>

namespace FlexFlow {

/**
 * @brief Finds the lowest common ancestor (LCA) of a set of nodes in a directed
 * graph.
 *
 * @details
 * Within this function, we consider the set of ancestors of a given node to
 * include the node itself, so the lowest common ancestor of a set of nodes can
 * be contained in the input set of nodes itself.
 *
 * For example, consider the following directed graph:
 *
 * ```
 * digraph {
 *   0 -> 1;
 *   0 -> 2;
 *   1 -> 3;
 *   1 -> 4;
 * }
 * ```
 *
 * The lowest common ancestor of nodes 3 and 1 is 1.
 *
 * @note
 * In a Directed Acyclic Graph, a set of nodes can have no LCA, a unique node as
 * LCA, or a set of nodes as LCA.
 */
std::optional<std::unordered_set<Node>>
    get_lowest_common_ancestors(DiGraphView const &g,
                                std::unordered_set<Node> const &nodes);

} // namespace FlexFlow

#endif
