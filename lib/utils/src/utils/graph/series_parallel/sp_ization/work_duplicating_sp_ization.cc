#include "utils/graph/series_parallel/sp_ization/work_duplicating_sp_ization.h"
#include "utils/containers/filter.h"
#include "utils/fmt/variant.h"
#include "utils/containers/get_only.h"
#include "utils/containers/group_by.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
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
#include <libassert/assert.hpp>
#include <unordered_set>

namespace FlexFlow {

static NonNormalSeriesSplit cut_off_head(NonNormalSeriesSplit const &s) {
  ASSERT(s.children.size() > 0);
  return NonNormalSeriesSplit{slice(s.children, 1, std::nullopt)};
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
  auto is_non_empty_strand = [](NonNormalSeriesSplit const &serial) {
    return !is_empty_non_normal(NonNormalSPDecomposition{serial});
  };

  auto strand_head = [](NonNormalSeriesSplit const &strand) {
    return strand.children.at(0);
  };

  std::unordered_set<NonNormalSeriesSplit> non_empty_strands =
      filter(strands, is_non_empty_strand);

  OneToMany<std::variant<NonNormalParallelSplit, Node>, NonNormalSeriesSplit>
      strands_grouped_by_head = group_by(non_empty_strands, strand_head);

  // recursively coalesce the strands
  std::unordered_multiset<NonNormalSPDecomposition> coalesced_strands;
  for (auto const &[head, strands_with_head] :
       strands_grouped_by_head.l_to_r()) {
    std::unordered_set<NonNormalSeriesSplit> tails =
        transform(strands_with_head, cut_off_head);
    NonNormalSPDecomposition parallel_comp =
        parallel_composition_with_coalescing(tails);

    NonNormalSPDecomposition head_as_decomposition =
        widen<NonNormalSPDecomposition>(head);
    NonNormalSPDecomposition series_comp =
        non_normal_series_composition({head_as_decomposition, parallel_comp});
    SeriesParallelDecomposition normalized =
        normalize_sp_decomposition(series_comp);
    coalesced_strands.insert(as_non_normal(normalized));
  }

  return non_normal_parallel_composition(coalesced_strands);
}

static SeriesParallelDecomposition
    work_duplicating_sp_ization_unchecked_with_coalescing(
        DiGraphView const &g) {
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
    work_duplicating_sp_ization_unchecked(DiGraphView const &g) {
  std::unordered_map<Node, NonNormalSPDecomposition> node_to_sp;

  for (Node const &node : get_topological_ordering(g)) {

    std::unordered_multiset<NonNormalSPDecomposition> predecessors_as_sp =
        unordered_multiset_of(
            transform(get_predecessors(g, node),
                      [&](Node const &p) { return node_to_sp.at(p); }));

    NonNormalSPDecomposition parallel_comp =
        non_normal_parallel_composition(predecessors_as_sp);
    NonNormalSPDecomposition node_as_decomposition =
        NonNormalSPDecomposition{node};
    NonNormalSPDecomposition sp_decomp =
        non_normal_series_composition({parallel_comp, node_as_decomposition});

    node_to_sp.emplace(node, sp_decomp);
  }

  Node sink = get_only(get_terminal_nodes(g));
  return normalize_sp_decomposition(node_to_sp.at(sink));
}

SeriesParallelDecomposition
    naive_work_duplicating_sp_ization(DiGraphView const &g) {
  ASSERT(is_2_terminal_dag(g));
  return work_duplicating_sp_ization_unchecked(g);
}

SeriesParallelDecomposition
    work_duplicating_sp_ization_with_coalescing(DiGraphView const &g) {
  ASSERT(is_2_terminal_dag(g));
  return work_duplicating_sp_ization_unchecked_with_coalescing(g);
}

} // namespace FlexFlow
