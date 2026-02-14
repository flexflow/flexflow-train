#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_NON_NORMAL_SP_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_NON_NORMAL_SP_DECOMPOSITION_H

#include "utils/graph/series_parallel/non_normal_sp_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <unordered_set>
#include <vector>

namespace FlexFlow {

bool is_empty_non_normal(NonNormalSPDecomposition const &sp);

NonNormalSPDecomposition non_normal_series_composition(
    std::vector<NonNormalSPDecomposition> const &sp_compositions);

NonNormalSPDecomposition non_normal_parallel_composition(
    std::unordered_multiset<NonNormalSPDecomposition> const &sp_compositions);

NonNormalSPDecomposition as_non_normal(SeriesParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
