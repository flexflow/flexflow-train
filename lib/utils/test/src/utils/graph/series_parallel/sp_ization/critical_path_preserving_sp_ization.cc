#include "utils/graph/series_parallel/sp_ization/critical_path_preserving_sp_ization.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/graph/series_parallel/series_parallel_splits.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("critical_path_preserving_sp_ization") {

    SUBCASE("Sample Graph #1") {
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

      std::unordered_map<Node, float> cost_map = {{n.at(0), 3},
                                                  {n.at(1), 2},
                                                  {n.at(2), 1},
                                                  {n.at(3), 1},
                                                  {n.at(4), 1},
                                                  {n.at(5), 5}};

      CHECK(work_cost(g, cost_map) == 13);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SeriesParallelDecomposition sp = critical_path_preserving_sp_ization(g);

      SUBCASE("structure") {
        Node sp0 = n.at(0);
        SeriesSplit sp1 = SeriesSplit{{sp0, n.at(1)}};
        SeriesSplit sp2 = SeriesSplit{{ParallelSplit{{sp0, sp1}}, n.at(2)}};
        SeriesSplit sp3 = SeriesSplit{{n.at(0), n.at(1), n.at(3)}};
        SeriesSplit sp4 = SeriesSplit{{ParallelSplit{{sp2, sp3}}, n.at(4)}};
        SeriesSplit sp5 = SeriesSplit{{ParallelSplit{{sp3, sp4}}, n.at(5)}};
        SeriesParallelDecomposition correct = SeriesParallelDecomposition{sp5};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = 3 * 4 + 2 * 3 + 1 * 1 + 1 * 2 + 1 * 1 + 5 * 1;
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #2") {
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

      std::unordered_map<Node, float> cost_map = {{n.at(0), 1},
                                                  {n.at(1), 1},
                                                  {n.at(2), 10},
                                                  {n.at(3), 1},
                                                  {n.at(4), 1},
                                                  {n.at(5), 1}};

      CHECK(work_cost(g, cost_map) == 15);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SeriesParallelDecomposition sp = critical_path_preserving_sp_ization(g);

      SUBCASE("structure") {
        SeriesParallelDecomposition correct = SeriesParallelDecomposition{
            SeriesSplit{{ParallelSplit{
                             {SeriesSplit{{n.at(0), n.at(1), n.at(3), n.at(4)}},
                              SeriesSplit{{n.at(0), n.at(2)}}}},
                         n.at(5)}}};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = 16;
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }
  }

  TEST_CASE("critical_path_preserving_sp_ization_with_coalescing") {

    SUBCASE("Sample Graph #1") {
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

      std::unordered_map<Node, float> cost_map = {{n.at(0), 1},
                                                  {n.at(1), 1},
                                                  {n.at(2), 2},
                                                  {n.at(3), 3},
                                                  {n.at(4), 1},
                                                  {n.at(5), 1}};

      CHECK(work_cost(g, cost_map) == 9);
      CHECK(critical_path_cost(g, cost_map) == 7);

      SeriesParallelDecomposition sp =
          critical_path_preserving_sp_ization_with_coalescing(g);

      SUBCASE("structure") {
        SeriesParallelDecomposition correct =
            SeriesParallelDecomposition{SeriesSplit{
                {n.at(0),
                 n.at(1),
                 ParallelSplit{
                     {SeriesSplit{{ParallelSplit{{n.at(2), n.at(3)}}, n.at(4)}},
                      n.at(3)}},
                 n.at(5)}}};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = 12;
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #2") {
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

      std::unordered_map<Node, float> cost_map = {{n.at(0), 1},
                                                  {n.at(1), 1},
                                                  {n.at(2), 10},
                                                  {n.at(3), 1},
                                                  {n.at(4), 1},
                                                  {n.at(5), 1}};

      CHECK(work_cost(g, cost_map) == 15);
      CHECK(critical_path_cost(g, cost_map) == 12);

      SeriesParallelDecomposition sp =
          critical_path_preserving_sp_ization_with_coalescing(g);

      SUBCASE("structure") {
        SeriesParallelDecomposition correct(SeriesSplit{
            {n.at(0),
             ParallelSplit{{SeriesSplit{{n.at(1), n.at(3), n.at(4)}}, n.at(2)}},
             n.at(5)}});
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      }
      SUBCASE("work cost") {
        float correct = 15;
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }

    SUBCASE("Sample Graph #3") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 10);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(3)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(1), n.at(5)},
                 DirectedEdge{n.at(1), n.at(4)},
                 DirectedEdge{n.at(2), n.at(6)},
                 DirectedEdge{n.at(3), n.at(4)},
                 DirectedEdge{n.at(3), n.at(5)},
                 DirectedEdge{n.at(3), n.at(8)},
                 DirectedEdge{n.at(4), n.at(8)},
                 DirectedEdge{n.at(5), n.at(7)},
                 DirectedEdge{n.at(7), n.at(8)},
                 DirectedEdge{n.at(6), n.at(9)},
                 DirectedEdge{n.at(8), n.at(9)}});

      std::unordered_map<Node, float> cost_map = {{n.at(0), 1},
                                                  {n.at(1), 1},
                                                  {n.at(2), 4},
                                                  {n.at(3), 10},
                                                  {n.at(4), 10},
                                                  {n.at(5), 5},
                                                  {n.at(6), 4},
                                                  {n.at(7), 3},
                                                  {n.at(8), 4},
                                                  {n.at(9), 1}};

      CHECK(work_cost(g, cost_map) == 43);
      CHECK(critical_path_cost(g, cost_map) == 26);

      SeriesParallelDecomposition sp =
          critical_path_preserving_sp_ization_with_coalescing(g);

      SUBCASE("structure") {

        SeriesParallelDecomposition correct =
            SeriesParallelDecomposition{SeriesSplit{
                {n.at(0),
                 ParallelSplit{
                     {SeriesSplit{{n.at(1), n.at(2), n.at(6)}},
                      SeriesSplit{
                          {ParallelSplit{
                               {SeriesSplit{
                                    {ParallelSplit{{n.at(1), n.at(3)}},
                                     ParallelSplit{
                                         {n.at(4),
                                          SeriesSplit{{n.at(5), n.at(7)}}}}}},
                                n.at(3)}},
                           n.at(8)}}}},
                 n.at(9)}}};
        SeriesParallelDecomposition result = sp;
        CHECK(correct == result);
      };
      SUBCASE("work cost") {
        float correct = 54;
        float result = work_cost(sp, cost_map);
        CHECK(correct == result);
      }

      SUBCASE("critical path cost") {
        float correct = critical_path_cost(g, cost_map);
        float result = critical_path_cost(sp, cost_map);
        CHECK(correct == result);
      }
    }
    SUBCASE("Transitive Reduction") {
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
          critical_path_preserving_sp_ization_with_coalescing(g);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{
              {n.at(0), n.at(1), ParallelSplit{{n.at(2), n.at(3)}}, n.at(4)}}};
      CHECK(result == correct);
    }
  }
}
