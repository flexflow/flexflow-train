#include "utils/graph/series_parallel/sp_ization/work_duplicating_spization.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/algorithms/is_2_terminal_dag.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/series_parallel/non_normal_sp_decomposition.h"
#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/variant.h"
#include <unordered_set>

namespace FlexFlow {

static NonNormalSeriesSplit cut_off_head(NonNormalSeriesSplit const &s) {
  assert(s.children.size() > 0);
  return NonNormalSeriesSplit{
      std::vector<std::variant<NonNormalParallelSplit, Node>>{
          s.children.begin() + 1, s.children.end()}};
}

/* Performs a parallel composition with coalescing, where components with a
 * common starting child are merged together
 * Example: to parallel compose S(1, 2, 5), S(1, 3, 4):
 *  without coalescing: P(S(1, 2, 5), S(1, 3, 4))
 *  with coalescing: S(1, P( S(2,5), S(3,4) ))
 */
static NonNormalSPDecomposition parallel_composition_with_coalescing(
    std::unordered_set<NonNormalSeriesSplit> const &strands) {
  if (strands.size() == 1) {
    return NonNormalSPDecomposition{get_only(strands)};
  }

  // group strands by their first ("head") node
  std::unordered_map<std::variant<NonNormalParallelSplit, Node>,
                     std::unordered_set<NonNormalSeriesSplit>>
      grouped_strands;
  for (NonNormalSeriesSplit predecessor :
       filter(strands, [](NonNormalSeriesSplit const &serial) {
         return !is_empty_non_normal(NonNormalSPDecomposition{serial});
       })) {
    grouped_strands[predecessor.children.at(0)].insert(
        cut_off_head(predecessor));
  }

  // recursively coalesce the strands
  std::unordered_multiset<NonNormalSPDecomposition> coalesced_strands;
  for (auto const &[head, tails] : grouped_strands) {
    NonNormalSPDecomposition parallel_comp =
        parallel_composition_with_coalescing(tails);

    NonNormalSPDecomposition series_comp = non_normal_series_composition(
        {widen<NonNormalSPDecomposition>(head), parallel_comp});
    coalesced_strands.insert(
        as_non_normal(normalize_sp_decomposition(series_comp)));
  }

  return non_normal_parallel_composition(coalesced_strands);
}

static SeriesParallelDecomposition
work_duplicating_spization_unchecked_with_coalescing(DiGraphView const &g) {
  std::unordered_map<Node, NonNormalSeriesSplit> node_to_sp;

  Node source = get_only(get_initial_nodes(g));
  node_to_sp.emplace(source, NonNormalSeriesSplit{{source}});

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::unordered_set<NonNormalSeriesSplit> predecessors_as_sp =
        transform(get_predecessors(g, node),
                  [&](Node const &p) { return node_to_sp.at(p); });

    NonNormalSPDecomposition parallel_composed_predecessors =
        as_non_normal(normalize_sp_decomposition(
            parallel_composition_with_coalescing(predecessors_as_sp)));
    NonNormalSeriesSplit sp_decomp =
        non_normal_series_composition(
            {parallel_composed_predecessors, NonNormalSPDecomposition{node}})
            .get<NonNormalSeriesSplit>();
    node_to_sp.emplace(node, sp_decomp);
  }

  Node sink = get_only(get_terminal_nodes(g));
  return normalize_sp_decomposition(
      NonNormalSPDecomposition{node_to_sp.at(sink)});
}

static SeriesParallelDecomposition
work_duplicating_spization_unchecked(DiGraphView const &g) {
  std::unordered_map<Node, NonNormalSPDecomposition> node_to_sp;

  for (Node const &node : get_topological_ordering(g)) {

    std::unordered_multiset<NonNormalSPDecomposition> predecessors_as_sp =
        unordered_multiset_of(
            transform(get_predecessors(g, node),
                      [&](Node const &p) { return node_to_sp.at(p); }));

    NonNormalSPDecomposition sp_decomp = non_normal_series_composition(
        {non_normal_parallel_composition(predecessors_as_sp),
         NonNormalSPDecomposition{node}});

    node_to_sp.emplace(node, sp_decomp);
  }

  Node sink = get_only(get_terminal_nodes(g));
  return normalize_sp_decomposition(node_to_sp.at(sink));
}

SeriesParallelDecomposition
naive_work_duplicating_spization(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return work_duplicating_spization_unchecked(g);
}

SeriesParallelDecomposition
work_duplicating_spization_with_coalescing(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return work_duplicating_spization_unchecked_with_coalescing(g);
}


} // namespace FlexFlow
