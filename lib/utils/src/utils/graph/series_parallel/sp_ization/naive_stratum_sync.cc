#include "utils/graph/series_parallel/sp_ization/naive_stratum_sync.h"
#include "utils/containers/maximum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/series_parallel/non_normal_sp_decomposition.h"
#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"

namespace FlexFlow {

std::vector<std::unordered_multiset<Node>>
stratum_split_assuming_unit_cost(DiGraphView const &g) {
  std::unordered_map<Node, nonnegative_int> node_to_stratum =
      get_longest_path_lengths_from_root(g);
  std::vector<std::unordered_multiset<Node>> result(
      maximum(values(node_to_stratum)).unwrap_nonnegative());
  for (auto const &[node, depth] : node_to_stratum) {
    result[depth.unwrap_nonnegative() - 1].insert(node);
  }
  return result;
}

static SeriesParallelDecomposition
naive_stratum_merge(std::vector<std::unordered_multiset<Node>> stratum_split) {
  std::vector<SeriesParallelDecomposition> strata = transform(
      stratum_split, [](std::unordered_multiset<Node> const &stratum_nodes) {
        return parallel_composition(transform(stratum_nodes, [](Node const &n) {
          return SeriesParallelDecomposition{n};
        }));
      });
  return normalize_sp_decomposition(as_non_normal(series_composition(strata)));
}

SeriesParallelDecomposition
stratum_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::vector<std::unordered_multiset<Node>> stratum_split =
      stratum_split_assuming_unit_cost(g);
  return naive_stratum_merge(stratum_split);
}

SeriesParallelDecomposition stratum_sync_sp_ization(DiGraphView const &g) {
  assert(is_acyclic(g));
  SeriesParallelDecomposition sp = stratum_sync_sp_ization_unchecked(g);
  assert(dependencies_are_maintained(g, sp));
  return sp;
}

} // namespace FlexFlow
