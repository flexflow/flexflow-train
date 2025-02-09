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
#include "utils/graph/series_parallel/extended_series_reduction.dtg.h"
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

  auto perform_extended_parallel_reduction =
      [&](ExtendedParallelReduction const &parallel_reduction) {
        MultiDiEdge merged =
            apply_extended_parallel_reduction(ttsp, parallel_reduction);

        SeriesParallelDecomposition new_tree = parallel_composition(transform(
            unordered_multiset_of(parallel_reduction.edges),
            [&](MultiDiEdge const &e) { return ttsp_edge_to_sp_tree.at(e); }));

        for (MultiDiEdge const &e : parallel_reduction.edges) {
          ttsp_edge_to_sp_tree.erase(e);
        }
        ttsp_edge_to_sp_tree.insert({merged, new_tree});

        return new_tree;
      };

  auto perform_extended_series_reduction =
      [&](ExtendedSeriesReduction const &series_reduction) {
        MultiDiEdge merged =
            apply_extended_series_reduction(ttsp, series_reduction);

        SeriesParallelDecomposition new_tree = series_composition(
            transform(series_reduction.edges, [&](MultiDiEdge const &e) {
              return ttsp_edge_to_sp_tree.at(e);
            }));

        for (MultiDiEdge const &e : series_reduction.edges) {
          ttsp_edge_to_sp_tree.erase(e);
        }
        ttsp_edge_to_sp_tree.insert({merged, new_tree});

        return new_tree;
      };

  while (true) {
    bool reduction_has_happened = false;

    std::unordered_set<ExtendedParallelReduction> parallel_reductions =
        find_all_extended_parallel_reductions(ttsp);

    if (!parallel_reductions.empty()) {
      for (ExtendedParallelReduction parallel_reduction : parallel_reductions) {
        perform_extended_parallel_reduction(parallel_reduction);
      }
      reduction_has_happened = true;
    }

    std::unordered_set<ExtendedSeriesReduction> series_reductions =
        find_all_extended_series_reductions(ttsp);
    if (!series_reductions.empty()) {
      for (ExtendedSeriesReduction series_reduction : series_reductions) {
        perform_extended_series_reduction(series_reduction);
      }
      reduction_has_happened = true;
    }

    if (reduction_has_happened) {
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

std::optional<SeriesParallelDecomposition>
    get_series_parallel_decomposition_unoptimized(DiGraphView const &g) {

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
  std::unordered_map<MultiDiEdge, BinarySPDecompositionTree>
      ttsp_edge_to_sp_tree = map_values(
          inverse_line_graph_result.inverse_edge_to_line_node_bidict
              .as_unordered_map(),
          [](Node const &n) { return BinarySPDecompositionTree{n}; });

  while (true) {
    assert(ttsp_edge_to_sp_tree.size() == get_edges(ttsp).size());
    std::optional<ParallelReduction> maybe_parallel_reduction =
        find_parallel_reduction(ttsp);
    if (maybe_parallel_reduction.has_value()) {
      ParallelReduction parallel_reduction = maybe_parallel_reduction.value();
      auto [e1, e2] = parallel_reduction.edges.ordered();
      MultiDiEdge merged = apply_parallel_reduction(ttsp, parallel_reduction);
      BinarySPDecompositionTree new_tree = BinarySPDecompositionTree{
          BinaryParallelSplit{
              ttsp_edge_to_sp_tree.at(e1),
              ttsp_edge_to_sp_tree.at(e2),
          },
      };
      ttsp_edge_to_sp_tree.erase(e1);
      ttsp_edge_to_sp_tree.erase(e2);
      ttsp_edge_to_sp_tree.insert({merged, new_tree});

      continue;
    }

    std::optional<SeriesReduction> maybe_series_reduction =
        find_series_reduction(ttsp);
    if (maybe_series_reduction.has_value()) {
      SeriesReduction series_reduction = maybe_series_reduction.value();
      MultiDiEdge e1 = series_reduction.first;
      MultiDiEdge e2 = series_reduction.second;
      MultiDiEdge merged = apply_series_reduction(ttsp, series_reduction);
      BinarySPDecompositionTree new_tree = BinarySPDecompositionTree{
          BinarySeriesSplit{
              ttsp_edge_to_sp_tree.at(e1),
              ttsp_edge_to_sp_tree.at(e2),
          },
      };
      ttsp_edge_to_sp_tree.erase(e1);
      ttsp_edge_to_sp_tree.erase(e2);
      ttsp_edge_to_sp_tree.insert({merged, new_tree});
      continue;
    }

    if (get_nodes(ttsp).size() != 2) {
      return std::nullopt;
    }
    if (get_edges(ttsp).size() != 1) {
      return std::nullopt;
    }

    MultiDiEdge e = get_only(get_edges(ttsp));
    if (ttsp.get_multidiedge_src(e) != ttsp.get_multidiedge_dst(e)) {
      return nary_sp_tree_from_binary(ttsp_edge_to_sp_tree.at(e));
    }
  }
}

} // namespace FlexFlow
