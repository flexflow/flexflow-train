#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_PARALLEL_DECOMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_SERIES_PARALLEL_DECOMPOSITION_H

#include "utils/graph/series_parallel/intermediate_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <variant>

namespace FlexFlow {

std::variant<SeriesSplit, ParallelSplit, Node> internal_to_final_ast(
    std::variant<IntermediateSpDecompositionTree, Node> const &ast);
SeriesParallelDecomposition
    to_final_ast(std::variant<IntermediateSpDecompositionTree, Node> const &);

std::unordered_multiset<Node> get_nodes(SeriesParallelDecomposition const &sp);
std::unordered_multiset<Node> get_nodes(SeriesSplit const &);
std::unordered_multiset<Node> get_nodes(ParallelSplit const &);
std::unordered_multiset<Node> get_nodes(Node const &);

bool is_empty(Node const &node);
bool is_empty(SeriesSplit const &serial);
bool is_empty(ParallelSplit const &parallel);
bool is_empty(SeriesParallelDecomposition const &sp);

bool has_no_duplicate_nodes(SeriesParallelDecomposition const &sp);

SeriesParallelDecomposition delete_node(SeriesParallelDecomposition sp,
                                        Node const &node);

// duplicate nodes within `sp` are counted multiple times
size_t num_nodes(SeriesParallelDecomposition const &sp);

SeriesParallelDecomposition series_composition(
    std::vector<SeriesParallelDecomposition> const &sp_compositions);
SeriesParallelDecomposition parallel_composition(
    std::unordered_multiset<SeriesParallelDecomposition> const
        &sp_compositions);

} // namespace FlexFlow

#endif
