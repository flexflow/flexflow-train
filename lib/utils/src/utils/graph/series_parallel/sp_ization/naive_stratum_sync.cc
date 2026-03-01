#include "utils/graph/series_parallel/sp_ization/naive_stratum_sync.h"
#include "utils/containers/group_by.h"
#include "utils/containers/keys.h"
#include "utils/containers/maximum.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/series_parallel/non_normal_sp_decomposition.h"
#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::vector<std::unordered_multiset<Node>>
    stratum_split_assuming_unit_cost(DiGraphView const &g) {
  std::unordered_map<Node, nonnegative_int> node_to_stratum =
      get_longest_path_lengths_from_root(g);

  std::unordered_set<Node> nodes = keys(node_to_stratum);
  OneToMany<nonnegative_int, Node> strata_to_nodes =
      group_by(nodes, [&](Node const &n) { return node_to_stratum.at(n); });

  nonnegative_int num_strata = maximum(strata_to_nodes.left_values());

  return transform(range(1, num_strata.unwrap_nonnegative() + 1),
                   [&](int depth) {
                     return unordered_multiset_of(
                         strata_to_nodes.at_l(nonnegative_int{depth}));
                   });
}

static SeriesParallelDecomposition naive_stratum_merge(
    std::vector<std::unordered_multiset<Node>> stratum_split) {

  auto merge_one_stratum =
      [&](std::unordered_multiset<Node> const &stratum_nodes) {
        auto as_singleton_sp = [](Node const &node) {
          return NonNormalSPDecomposition{node};
        };
        return non_normal_parallel_composition(
            transform(stratum_nodes, as_singleton_sp));
      };

  std::vector<NonNormalSPDecomposition> parallel_strata =
      transform(stratum_split, merge_one_stratum);

  NonNormalSPDecomposition strata_in_series_order =
      non_normal_series_composition(parallel_strata);
  return normalize_sp_decomposition(strata_in_series_order);
}

SeriesParallelDecomposition
    naive_stratum_sync_sp_ization_unchecked(DiGraphView const &g) {

  std::vector<std::unordered_multiset<Node>> stratum_split =
      stratum_split_assuming_unit_cost(g);
  return naive_stratum_merge(stratum_split);
}

SeriesParallelDecomposition
    naive_stratum_sync_sp_ization(DiGraphView const &g) {
  ASSERT(is_acyclic(g));
  SeriesParallelDecomposition sp = naive_stratum_sync_sp_ization_unchecked(g);
  ASSERT(dependencies_are_maintained(g, sp));
  return sp;
}

} // namespace FlexFlow
