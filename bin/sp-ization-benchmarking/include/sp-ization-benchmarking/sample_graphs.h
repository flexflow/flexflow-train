#ifndef _FLEXFLOW_BIN_SP_IZATION_BENCHMARKING_INCLUDE_SP_IZATION_BENCHMARKING_SAMPLE_GRAPHS_H
#define _FLEXFLOW_BIN_SP_IZATION_BENCHMARKING_INCLUDE_SP_IZATION_BENCHMARKING_SAMPLE_GRAPHS_H

#include "utils/graph/digraph/digraph.h"
#include <tuple>

namespace FlexFlow {

std::tuple<DiGraph, Node, Node> make_normal_taso_nasnet_cell();
std::tuple<DiGraph, Node, Node> make_reduction_taso_nasnet_cell();
DiGraph make_full_taso_nasnet(size_t num_reduction_cells, size_t N);
DiGraph make_linear(size_t length);
DiGraph make_rhombus();
DiGraph make_diamond();
DiGraph make_fully_connected(std::vector<size_t> const &layer_sizes);
DiGraph make_parallel_chains(size_t chain_length, size_t chain_num);
DiGraph make_sample_dag_1();
DiGraph make_sample_dag_2();
DiGraph make_sample_dag_3();
DiGraph make_taso_nasnet_cell();
DiGraph make_2_terminal_random_dag(size_t num_nodes, float p, size_t step);

} // namespace FlexFlow

#endif
