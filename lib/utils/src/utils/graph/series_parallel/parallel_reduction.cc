#include "utils/graph/series_parallel/parallel_reduction.h"
#include "utils/commutative_pair.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_one_of.h"
#include "utils/containers/group_by.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/multidigraph/algorithms/get_directed_edge.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/multidigraph/multidigraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/extended_parallel_reduction.dtg.h"
#include "utils/hash/unordered_set.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

ParallelReduction make_parallel_reduction(MultiDiEdge const &e1,
                                          MultiDiEdge const &e2) {
  return ParallelReduction{commutative_pair{e1, e2}};
}

std::optional<ParallelReduction>
    find_parallel_reduction(MultiDiGraphView const &g) {

  std::unordered_map<DirectedEdge, MultiDiEdge> seen;
  for (MultiDiEdge const &edge : get_edges(g)) {
    DirectedEdge diedge = get_directed_edge(g, edge);
    if (contains_key(seen, diedge)) {
      return make_parallel_reduction(seen.at(diedge), edge);
    }
    seen.emplace(diedge, edge);
  }
  return std::nullopt;
}

std::unordered_set<ExtendedParallelReduction>
    find_all_extended_parallel_reductions(MultiDiGraphView const &g) {
  std::unordered_map<DirectedEdge, std::unordered_set<MultiDiEdge>>
      reduction_groups;
  for (MultiDiEdge const &edge : get_edges(g)) {
    reduction_groups[get_directed_edge(g, edge)].insert(edge);
  }

  std::unordered_set<std::unordered_set<MultiDiEdge>> reductions = filter(
      unordered_set_of(values(reduction_groups)),
      [](std::unordered_set<MultiDiEdge> const &s) { return s.size() > 1; });

  return transform(reductions,
                   [&](std::unordered_set<MultiDiEdge> const &edges) {
                     return ExtendedParallelReduction{edges};
                   });
}

MultiDiEdge apply_parallel_reduction(MultiDiGraph &g,
                                     ParallelReduction const &r) {
  g.remove_edge(r.edges.max());
  return r.edges.min();
}

MultiDiEdge apply_extended_parallel_reduction(
    MultiDiGraph &g, ExtendedParallelReduction const &reduction) {

  MultiDiEdge keep_edge = get_one_of(reduction.edges);

  for (MultiDiEdge const &parallel_edge : reduction.edges) {
    if (parallel_edge != keep_edge) {
      g.remove_edge(parallel_edge);
    }
  }

  return keep_edge;
}

} // namespace FlexFlow
