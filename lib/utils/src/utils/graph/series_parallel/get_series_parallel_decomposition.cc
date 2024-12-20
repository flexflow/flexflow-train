#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/nary_sp_tree_from_binary.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/graph/series_parallel/parallel_reduction.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/graph/series_parallel/series_reduction.h"

namespace FlexFlow {

std::optional<SeriesParallelDecomposition>
    get_series_parallel_decomposition(DiGraphView const &g) {

  DiGraphView transitively_reduced = transitive_reduction(g);

  InverseLineGraphResult inverse_line_graph_result = ({
    std::optional<InverseLineGraphResult> maybe_line_graph =
        get_inverse_line_graph(transitively_reduced);
    if (!maybe_line_graph.has_value()) {
      return std::nullopt;
    }
    maybe_line_graph.value();
  });

  MultiDiGraph ttsp = MultiDiGraph::materialize_copy_of<AdjacencyMultiDiGraph>(
      inverse_line_graph_result.graph);

  std::unordered_map<MultiDiEdge, SeriesParallelDecomposition>
      ttsp_edge_to_sp_tree = map_values(
          inverse_line_graph_result.inverse_edge_to_line_node_bidict
              .as_unordered_map(),
          [](Node const &n) { return SeriesParallelDecomposition{n}; });

  while (true) {
    int reductions = 0;

    std::unordered_map<DirectedEdge, std::unordered_set<MultiDiEdge>>
        parallel_reductions = find_all_extended_parallel_reductions(ttsp);

    if (!parallel_reductions.empty()) {
      for (auto const &[_, parallel_reduction] : parallel_reductions) {
        MultiDiEdge merged =
            apply_extended_parallel_reduction(ttsp, parallel_reduction);

        SeriesParallelDecomposition new_tree = parallel_composition(transform(
            unordered_multiset_of(parallel_reduction),
            [&](MultiDiEdge const &e) { return ttsp_edge_to_sp_tree.at(e); }));

        for (MultiDiEdge const &e : parallel_reduction) {
          ttsp_edge_to_sp_tree.erase(e);
        }
        ttsp_edge_to_sp_tree.insert({merged, new_tree});
      }
      reductions++;
    }

    std::unordered_set<std::vector<MultiDiEdge>> series_reductions =
        find_all_extended_series_reductions(ttsp);
    if (!series_reductions.empty()) {
      for (std::vector<MultiDiEdge> series_reduction : series_reductions) {
        MultiDiEdge merged =
            apply_extended_series_reduction(ttsp, series_reduction);

        SeriesParallelDecomposition new_tree = serial_composition(
            transform(series_reduction, [&](MultiDiEdge const &e) {
              return ttsp_edge_to_sp_tree.at(e);
            }));

        for (MultiDiEdge const &e : series_reduction) {
          ttsp_edge_to_sp_tree.erase(e);
        }
        ttsp_edge_to_sp_tree.insert({merged, new_tree});
      }
      reductions++;
    }

    if (reductions > 0) {
      continue;
    }

    if (get_nodes(ttsp).size() != 2 || get_edges(ttsp).size() != 1) {
      return std::nullopt;
    }

    MultiDiEdge e = get_only(get_edges(ttsp));
    if (ttsp.get_multidiedge_src(e) != ttsp.get_multidiedge_dst(e)) {
      return ttsp_edge_to_sp_tree.at(e);
    }
  }
}

} // namespace FlexFlow
