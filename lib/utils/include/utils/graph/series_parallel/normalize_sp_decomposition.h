#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_normalize_sp_decomposition_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_normalize_sp_decomposition_H

#include "utils/graph/series_parallel/non_normal_sp_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

namespace FlexFlow {

/**
 * @brief Normalizes a series-parallel decomposition into its canonical form.
 *
 * @details A normalized series-parallel decomposition satisfies the following
 * invariants:
 * - No empty SeriesSplit or ParallelSplit nodes (i.e., nodes with zero children)
 * - No SeriesSplit or ParallelSplit nodes with exactly one child
 *   (these are replaced by their child)
 *
 * These invariants ensure a unique canonical representation for any given
 * series-parallel structure.
 *
 * Examples:
 * - <tt>S(P(S()), Node(1), Node(2)) -> S(Node(1), Node(2))</tt>
 * - <tt>S(S(Node(1)), P(Node(2))) -> S(Node(1), Node(2))</tt>
 *
 */
SeriesParallelDecomposition
    normalize_sp_decomposition(NonNormalSPDecomposition const &sp);

} // namespace FlexFlow

#endif
