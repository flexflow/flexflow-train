#include "utils/graph/series_parallel/sp_ization/naive_stratum_sync.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/series_parallel/parallel_split.dtg.h"
#include "utils/graph/series_parallel/series_parallel_metrics.h"
#include "utils/graph/series_parallel/series_split.dtg.h"
#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("naive_stratum_sync_sp_ization") {

    SUBCASE("fully parallel graph becomes one parallel stratum") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);

      std::unordered_map<Node, float> cost_map = {
          {n.at(0), 1.0f}, {n.at(1), 5.0f}, {n.at(2), 2.0f}, {n.at(3), 3.0f}};

      SeriesParallelDecomposition sp = naive_stratum_sync_sp_ization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          ParallelSplit{{n.at(0), n.at(1), n.at(2), n.at(3)}}};
      CHECK(sp == correct);
      CHECK(work_cost(sp, cost_map) == work_cost(g, cost_map));
      CHECK(critical_path_cost(sp, cost_map) ==
            critical_path_cost(g, cost_map));
      CHECK(dependencies_are_maintained(g, sp));
    }

    SUBCASE("fully serial chain stays fully serial") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 5);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(3), n.at(4)},
                });

      std::unordered_map<Node, float> cost_map = {{n.at(0), 1.0f},
                                                  {n.at(1), 2.0f},
                                                  {n.at(2), 3.0f},
                                                  {n.at(3), 4.0f},
                                                  {n.at(4), 5.0f}};

      SeriesParallelDecomposition sp = naive_stratum_sync_sp_ization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n.at(0), n.at(1), n.at(2), n.at(3), n.at(4)}}};
      CHECK(sp == correct);
      CHECK(work_cost(sp, cost_map) == work_cost(g, cost_map));
      CHECK(critical_path_cost(sp, cost_map) ==
            critical_path_cost(g, cost_map));
      CHECK(dependencies_are_maintained(g, sp));
    }

    SUBCASE("two-layer graph with all-to-all connection") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 5);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(0), n.at(4)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(1), n.at(4)},
                });

      std::unordered_map<Node, float> cost_map = {{n.at(0), 2.0f},
                                                  {n.at(1), 3.0f},
                                                  {n.at(2), 5.0f},
                                                  {n.at(3), 7.0f},
                                                  {n.at(4), 11.0f}};

      SeriesParallelDecomposition sp = naive_stratum_sync_sp_ization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{{n.at(0), n.at(1)}},
                       ParallelSplit{{n.at(2), n.at(3), n.at(4)}}}}};
      CHECK(sp == correct);
      CHECK(work_cost(sp, cost_map) == work_cost(g, cost_map));
      CHECK(critical_path_cost(sp, cost_map) ==
            critical_path_cost(g, cost_map));
      CHECK(dependencies_are_maintained(g, sp));
    }

    SUBCASE("barrier synchronization can increase critical path while "
            "preserving work") {
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

      std::unordered_map<Node, float> cost_map = {{n.at(0), 1.0f},
                                                  {n.at(1), 1.0f},
                                                  {n.at(2), 10.0f},
                                                  {n.at(3), 1.0f},
                                                  {n.at(4), 1.0f},
                                                  {n.at(5), 1.0f}};

      SeriesParallelDecomposition sp = naive_stratum_sync_sp_ization(g);

      CHECK(work_cost(sp, cost_map) == work_cost(g, cost_map));
      CHECK(critical_path_cost(sp, cost_map) > critical_path_cost(g, cost_map));
      CHECK(critical_path_cost(g, cost_map) == 12.0f);
      CHECK(critical_path_cost(sp, cost_map) == 14.0f);
      CHECK(dependencies_are_maintained(g, sp));
    }

    SUBCASE("diamond-with-tail stratifies into four explicit layers") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(5)},
                    DirectedEdge{n.at(4), n.at(5)},
                });

      SeriesParallelDecomposition sp = naive_stratum_sync_sp_ization(g);

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n.at(0),
                       ParallelSplit{{n.at(1), n.at(2)}},
                       ParallelSplit{{n.at(3), n.at(4)}},
                       n.at(5)}}};
      CHECK(sp == correct);
      CHECK(dependencies_are_maintained(g, sp));
    }
  }
}
