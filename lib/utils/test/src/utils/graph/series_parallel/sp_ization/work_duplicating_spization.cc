#include "utils/graph/series_parallel/sp_ization/work_duplicating_spization.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/generate_map.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/parallel_split.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/graph/series_parallel/series_split.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

static std::pair<DiGraph, std::unordered_map<Node, float>>
    generate_random_2_terminal_weighted_dag(int max_num_nodes = 10,
                                            int max_num_edges = 20) {
  assert(max_num_nodes >= 2);

  int num_nodes = *rc::gen::inRange(2, max_num_nodes);

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> nodes = add_nodes(g, num_nodes);
  Node source = nodes.front();
  Node sink = nodes.back();

  int num_edges = *rc::gen::inRange(0, max_num_edges + 1);
  for (int i = 0; i < num_edges; i++) {
    int src_idx = *rc::gen::inRange(0, num_nodes - 1);
    int dst_idx = *rc::gen::inRange(src_idx + 1, num_nodes);
    g.add_edge(DirectedEdge{nodes.at(src_idx), nodes.at(dst_idx)});
  }

  for (Node const &n : get_initial_nodes(g)) {
    if (n != source) {
      g.add_edge(DirectedEdge{source, n});
    }
  }
  for (Node const &n : get_terminal_nodes(g)) {
    if (n != sink) {
      g.add_edge(DirectedEdge{n, sink});
    }
  }

  std::unordered_map<Node, float> cost_map =
      generate_map(get_nodes(g), [](Node const &) {
        return static_cast<float>(*rc::gen::inRange(1, 101));
      });

  return {g, cost_map};
}

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("naive_work_duplicating_spization") {

    SUBCASE("linear chain") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                });

      SeriesParallelDecomposition result =
          naive_work_duplicating_spization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n.at(0), n.at(1), n.at(2)}}};
      CHECK(correct == result);
    }

    SUBCASE("diamond") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                });

      SeriesParallelDecomposition result =
          naive_work_duplicating_spization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{{SeriesSplit{{n.at(0), n.at(1)}},
                                     SeriesSplit{{n.at(0), n.at(2)}}}},
                       n.at(3)}}};
      CHECK(correct == result);
    }

    SUBCASE("parallel paths of different lengths") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(5)},
                    DirectedEdge{n.at(3), n.at(4)},
                    DirectedEdge{n.at(4), n.at(5)},
                });

      SeriesParallelDecomposition result =
          naive_work_duplicating_spization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{
                           {SeriesSplit{{n.at(0), n.at(1), n.at(3), n.at(4)}},
                            SeriesSplit{{n.at(0), n.at(2)}}}},
                       n.at(5)}}};
      CHECK(correct == result);
    }

    RC_SUBCASE("critical path cost is preserved",
                []() {
                  auto [g, cost_map] =
                      generate_random_2_terminal_weighted_dag();
                  SeriesParallelDecomposition sp =
                      naive_work_duplicating_spization(g);
                  float original_cost = critical_path_cost(g, cost_map);
                  float sp_cost = critical_path_cost(sp, cost_map);
                  RC_ASSERT(original_cost == sp_cost);
                });
  }

  TEST_CASE("work_duplicating_spization_with_coalescing") {

    SUBCASE("diamond") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                });

      SeriesParallelDecomposition result =
          work_duplicating_spization_with_coalescing(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{
              {n.at(0), ParallelSplit{{n.at(1), n.at(2)}}, n.at(3)}}};
      CHECK(correct == result);
    }

    SUBCASE("parallel paths of different lengths") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(5)},
                    DirectedEdge{n.at(3), n.at(4)},
                    DirectedEdge{n.at(4), n.at(5)},
                });

      SeriesParallelDecomposition result =
          work_duplicating_spization_with_coalescing(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{
              {n.at(0),
               ParallelSplit{
                   {SeriesSplit{{n.at(1), n.at(3), n.at(4)}}, n.at(2)}},
               n.at(5)}}};
      CHECK(correct == result);
    }

    SUBCASE("parallel strands with cross edges") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(4)},
                    DirectedEdge{n.at(3), n.at(5)},
                    DirectedEdge{n.at(4), n.at(5)},
                });

      SeriesParallelDecomposition result =
          work_duplicating_spization_with_coalescing(g);

      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{
              {n.at(0),
               n.at(1),
               ParallelSplit{
                   {SeriesSplit{{ParallelSplit{{n.at(2), n.at(3)}}, n.at(4)}},
                    n.at(3)}},
               n.at(5)}}};
      CHECK(correct == result);
    }

    SUBCASE("graph with transitive edges") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 5);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(4)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(1), n.at(4)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(4)},
                });

      SeriesParallelDecomposition result =
          work_duplicating_spization_with_coalescing(g);

      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{
              {n.at(0), n.at(1), ParallelSplit{{n.at(2), n.at(3)}}, n.at(4)}}};
      CHECK(correct == result);
    }

    RC_SUBCASE("critical path cost is preserved",
                []() {
                  auto [g, cost_map] =
                      generate_random_2_terminal_weighted_dag();
                  SeriesParallelDecomposition sp =
                      work_duplicating_spization_with_coalescing(g);
                  float original_cost = critical_path_cost(g, cost_map);
                  float sp_cost = critical_path_cost(sp, cost_map);
                  RC_ASSERT(original_cost == sp_cost);
                });
  }
}
