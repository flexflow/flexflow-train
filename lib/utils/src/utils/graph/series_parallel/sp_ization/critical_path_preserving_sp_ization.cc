#include "utils/graph/series_parallel/sp_ization/critical_path_preserving_sp_ization.h"
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
#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/variant.h"
#include <unordered_set>

namespace FlexFlow {

static SeriesSplit cut_off_head(SeriesSplit const &s) {
  assert(s.children.size() > 0);
  return SeriesSplit{std::vector<std::variant<ParallelSplit, Node>>(
      s.children.begin() + 1, s.children.end())};
}

/* Performs a parallel composition with coalescing, where components with a
 * common starting child are merged together
 * Example: to parallel compose S(1, 2, 5), S(1, 3, 4):
 *  without coalescing: P(S(1, 2, 5), S(1, 3, 4))
 *  with coalescing: S(1, P( S(2,5), S(3,4) ))
 */
static SeriesParallelDecomposition parallel_composition_with_coalescing(
    std::unordered_set<SeriesSplit> const &strands) {
  if (strands.size() == 1) {
    return SeriesParallelDecomposition(get_only(strands));
  }

  // group strands by their first ("head") node
  std::unordered_map<std::variant<ParallelSplit, Node>,
                     std::unordered_set<SeriesSplit>>
      grouped_strands;
  for (SeriesSplit predecessor : filter(strands, [](SeriesSplit const &serial) {
         return !is_empty(serial);
       })) {
    grouped_strands[predecessor.children.at(0)].insert(
        cut_off_head(predecessor));
  }

  // recursively coalesce the strands
  std::unordered_multiset<SeriesParallelDecomposition> coalesced_strands;
  for (auto const &[head, tails] : grouped_strands) {
    SeriesParallelDecomposition parallel_comp =
        parallel_composition_with_coalescing(tails);
    coalesced_strands.insert(series_composition(
        {widen<SeriesParallelDecomposition>(head), parallel_comp}));
  }

  return normalize_sp_decomposition(parallel_composition(coalesced_strands));
}

static SeriesParallelDecomposition
    critical_path_preserving_sp_ization_unchecked_with_coalescing(
        DiGraphView const &g) {
  std::unordered_map<Node, SeriesSplit> node_to_sp;

  Node source = get_only(get_initial_nodes(g));
  node_to_sp.emplace(source, SeriesSplit{{source}});

  for (Node const &node : get_topological_ordering(g)) {
    if (node == source) {
      continue;
    }
    std::unordered_set<SeriesSplit> predecessors_as_sp =
        transform(get_predecessors(g, node),
                  [&](Node const &p) { return node_to_sp.at(p); });

    SeriesParallelDecomposition parallel_composed_predecessors =
        parallel_composition_with_coalescing(predecessors_as_sp);
    SeriesParallelDecomposition sp_decomp = series_composition(
        {parallel_composed_predecessors, SeriesParallelDecomposition(node)});
    node_to_sp.emplace(node, sp_decomp.get<SeriesSplit>());
  }

  Node sink = get_only(get_terminal_nodes(g));
  return normalize_sp_decomposition(
      SeriesParallelDecomposition(node_to_sp.at(sink)));
}

SeriesParallelDecomposition
    critical_path_preserving_sp_ization_with_coalescing(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return critical_path_preserving_sp_ization_unchecked_with_coalescing(g);
}

static SeriesParallelDecomposition
    critical_path_preserving_sp_ization_unchecked(DiGraphView const &g) {
  std::unordered_map<Node, SeriesParallelDecomposition> node_to_sp;

  for (Node const &node : get_topological_ordering(g)) {

    std::unordered_multiset<SeriesParallelDecomposition> predecessors_as_sp =
        unordered_multiset_of(
            transform(get_predecessors(g, node),
                      [&](Node const &p) { return node_to_sp.at(p); }));

    SeriesParallelDecomposition sp_decomp = series_composition(
        {normalize_sp_decomposition(parallel_composition(predecessors_as_sp)),
         SeriesParallelDecomposition(node)});

    node_to_sp.emplace(node, sp_decomp);
  }

  Node sink = get_only(get_terminal_nodes(g));
  return node_to_sp.at(sink);
}

SeriesParallelDecomposition
    critical_path_preserving_sp_ization(DiGraphView const &g) {
  assert(is_2_terminal_dag(g));
  return critical_path_preserving_sp_ization_unchecked(g);
}

} // namespace FlexFlow
