#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLLEL_GET_ANCESTORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLLEL_GET_ANCESTORS_H

#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

/**
 * @brief For a given node \p node and a series-parallel decomposition \p sp,
 * returns the set of nodes that are ancestors of the given node. We require
 * that \p node is a node in \p sp.
 *
 * @details The ancestors are
 * computed recursively as follows:
 *
 * If \p sp is a single node, then the ancestors are always empty.
 *
 * If \p sp = S(a_1, ..., a_n), where a_i are series-parallel structures,
 * and suppose that the node \p node is in a_j. Then:
 * - All the nodes in a_1, ..., a_{j-1} are ancestors of \p node.
 * - There is some subset of nodes of a_j that are ancestors of \p node
 *   (which we recursively compute).
 * - The nodes in a_{j+1}, ..., a_n are NOT ancestors of \p node.
 *
 * If \p sp = P(a_1, ..., a_n), where a_i are series-parallel structures,
 * then there is exactly one branch a_j of \p sp that contains \p node.
 * All the other branches are not ancestors of \p node (since they are
 * parallel to it). So we recursively compute the ancestors of \p node
 * within a_j.
 *
 * @example
 * For sp = S(n0, P(S(n1, n2), n3), n4, n5):
 *
 *   node | ancestors
 *   -----|----------
 *   n0   | {}
 *   n1   | {n0}
 *   n2   | {n0, n1}
 *   n3   | {n0}
 *   n4   | {n0, n1, n2, n3}
 *   n5   | {n0, n1, n2, n3, n4}
 *
 */
std::unordered_set<Node> get_ancestors(SeriesParallelDecomposition const &sp,
                                       Node const &node);

} // namespace FlexFlow

#endif
