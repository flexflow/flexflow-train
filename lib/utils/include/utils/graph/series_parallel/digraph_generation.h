#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_DIGRAPH_GENERATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_DIGRAPH_GENERATION_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

std::unordered_map<Node, Node> parallel_extend(DiGraph &g,
                                               DiGraphView const &ext);
std::unordered_map<Node, Node> serial_extend(DiGraph &g,
                                             DiGraphView const &ext);
DiGraph series_composition(DiGraphView const &g1, DiGraphView const &g2);
DiGraph parallel_composition(DiGraphView const &g1, DiGraphView const &g2);
DiGraph series_composition(std::vector<DiGraphView> const &graphs);
DiGraph parallel_composition(std::vector<DiGraphView> const &graphs);

/**
 * @brief Constructs a directed DiGraph from a series-parallel decomposition.
 *
 * @details The transformation is performed recursively as follows:
 * - Nodes in the decomposition remain the same in the resulting graph (but the node ids are fresh)
 * - For serial composition between graphs, an all-to-all connection is created
 * between the terminal nodes of one graph and the initial nodes of the
 * following one.
 * - For parallel composition between graphs, the union of the graphs is taken
 * without adding any additional edges.
 *
 */
DiGraph digraph_from_sp_decomposition(SeriesParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
