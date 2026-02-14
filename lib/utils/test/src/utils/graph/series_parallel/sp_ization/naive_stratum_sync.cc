#include "utils/graph/series_parallel/sp_ization/naive_stratum_sync.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/series_parallel/parallel_split.dtg.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/graph/series_parallel/series_split.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("naive_stratum_sync") {

    SUBCASE("Sample Graph #1") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[4]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[4], n[5]},
                });

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1}, {n[1], 1}, {n[2], 2}, {n[3], 3}, {n[4], 1}, {n[5], 1}};

      CHECK(work_cost(g, cost_map) == 9);
      CHECK(critical_path_cost(g, cost_map) == 7);

      SeriesParallelDecomposition sp = stratum_sync_sp_ization(g);

      SUBCASE("structure") {
        SeriesParallelDecomposition correct = SeriesParallelDecomposition{
            SeriesSplit{{n[0], n[1], ParallelSplit{{n[2], n[3]}}, n[4], n[5]}}};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 7;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #2") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[5]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[4], n[5]},
                });

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1}, {n[1], 1}, {n[2], 10}, {n[3], 1}, {n[4], 1}, {n[5], 1}};

      CHECK(work_cost(g, cost_map) == 15);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SeriesParallelDecomposition sp = stratum_sync_sp_ization(g);

      SUBCASE("structure") {
        SeriesParallelDecomposition correct = SeriesParallelDecomposition{
            SeriesSplit{{n[0], ParallelSplit{{n[1], n[2]}}, n[3], n[4], n[5]}}};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 14;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #3") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 9);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[3]},
                    DirectedEdge{n[1], n[2]},
                    DirectedEdge{n[1], n[5]},
                    DirectedEdge{n[1], n[4]},
                    DirectedEdge{n[2], n[6]},
                    DirectedEdge{n[3], n[4]},
                    DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[3], n[8]},
                    DirectedEdge{n[4], n[8]},
                    DirectedEdge{n[5], n[7]},
                    DirectedEdge{n[7], n[8]},
                });

      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 1},
                                                  {n[2], 10},
                                                  {n[3], 10},
                                                  {n[4], 1},
                                                  {n[5], 1},
                                                  {n[6], 10},
                                                  {n[7], 10},
                                                  {n[8], 1}};

      CHECK(work_cost(g, cost_map) == 45);
      CHECK(critical_path_cost(g, cost_map) == 23);

      SeriesParallelDecomposition sp = stratum_sync_sp_ization(g);

      SUBCASE("structure") {
        SeriesParallelDecomposition correct = SeriesParallelDecomposition{
            SeriesSplit{{n[0],
                         ParallelSplit{{n[1], n[3]}},
                         ParallelSplit{{n[2], n[4], n[5]}},
                         ParallelSplit{{n[6], n[7]}},
                         n[8]}}};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = work_cost(g, cost_map);
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = 32;
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }
  }
}

