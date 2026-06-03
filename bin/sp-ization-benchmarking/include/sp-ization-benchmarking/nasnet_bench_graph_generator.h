/**
 * @brief Utilities for generating random DAGs based on the NASNet-A
 * architecture. NASNet-A is composed of a series of cells, which we randomly
 * generate.
 *
 * For context, see:
 * - Paper: https://arxiv.org/abs/1902.09635
 * - Reference implementation:
 * https://github.com/google-research/nasbench/blob/b94247037ee470418a3e56dcb83814e9be83f3a8/nasbench/api.py
 */

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <optional>
#include <vector>

namespace FlexFlow {

struct NasNetBenchConfig {
  std::vector<std::vector<bool>> adjacency_matrix;
};

bool is_valid_config(NasNetBenchConfig const &config);

bool is_valid_cell(DiGraphView const &g);

NasNetBenchConfig generate_random_config();

std::optional<DiGraph>
    maybe_generate_nasnet_bench_cell(NasNetBenchConfig const &config);

DiGraph generate_nasnet_bench_cell();

DiGraph generate_nasnet_bench_network();

} // namespace FlexFlow
