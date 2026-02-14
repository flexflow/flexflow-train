#include "utils/graph/series_parallel/sp_ization/flexible_algo.h"
#include "utils/containers/values.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node.dtg.h"
#include "utils/graph/series_parallel/parallel_split.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_split.dtg.h"
#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"
#include "utils/graph/series_parallel/sp_ization/node_role.dtg.h"
#include <doctest/doctest.h>
#include <unordered_map>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flexible_algo") {
    SUBCASE("Single Node") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      Node n0 = g.add_node();

      std::unordered_map<Node, float> cost_map = {{n0, 1.0f}};

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n0};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("Tri Node Graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[2], n[3]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 1.0f},
          {n[3], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0], ParallelSplit{{n[1], n[2]}}, n[3]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("Series") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[1], n[2]},
                 DirectedEdge{n[2], n[3]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 1.0f},
          {n[3], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n[0], n[1], n[2], n[3]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("6 Node Diamond Graph - constant cost map") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[4], n[5]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 1.0f},
          {n[3], 1.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n[0],
                                                   ParallelSplit{{n[1], n[2]}},
                                                   ParallelSplit{{n[3], n[4]}},
                                                   n[5]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("6 Node Diamond Graph - cost map v2") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[4], n[5]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 10.0f},
          {n[3], 1000.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       n[1],
                       ParallelSplit{{SeriesSplit{{n[2], n[4]}}, n[3]}},
                       n[5]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("6 Node Diamond Graph - cost map v3") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[4], n[5]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 1000.0f},
          {n[3], 1000.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       ParallelSplit{{SeriesSplit{{n[1], n[3]}}, n[2]}},
                       n[4],
                       n[5]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("With Parallel Strand") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 8);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[4], n[6]},
                 DirectedEdge{n[5], n[7]},
                 DirectedEdge{n[6], n[7]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 1.0f},
          {n[3], 1.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
          {n[6], 1.0f},
          {n[7], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       ParallelSplit{{SeriesSplit{{n[1], n[3], n[5]}},
                                      SeriesSplit{{n[2], n[4], n[6]}}}},
                       n[7]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("Simple With Parallel Strand") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 7);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[6]},
                 DirectedEdge{n[4], n[6]},
                 DirectedEdge{n[0], n[5]},
                 DirectedEdge{n[5], n[6]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 100.0f},
          {n[3], 100.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
          {n[6], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       ParallelSplit{
                           {SeriesSplit{{ParallelSplit{
                                             {SeriesSplit{{n[1], n[3]}}, n[2]}},
                                         n[4]}},
                            n[5]}},
                       n[6]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("With Appendage - constant cost") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 8);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[4], n[5]},
                 DirectedEdge{n[0], n[6]},
                 DirectedEdge{n[6], n[5]},
                 DirectedEdge{n[2], n[7]},
                 DirectedEdge{n[7], n[5]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 1.0f},
          {n[2], 1.0f},
          {n[3], 1.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
          {n[6], 1.0f},
          {n[7], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       ParallelSplit{
                           {n[6],
                            SeriesSplit{{ParallelSplit{{n[1], n[2]}},
                                         ParallelSplit{{n[3], n[4], n[7]}}}}}},
                       n[5]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("With Appendage - weighted") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 8);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[1], n[4]},
                 DirectedEdge{n[2], n[4]},
                 DirectedEdge{n[3], n[5]},
                 DirectedEdge{n[4], n[5]},
                 DirectedEdge{n[0], n[6]},
                 DirectedEdge{n[6], n[5]},
                 DirectedEdge{n[2], n[7]},
                 DirectedEdge{n[7], n[5]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 100.0f},
          {n[2], 1.0f},
          {n[3], 1.0f},
          {n[4], 1.0f},
          {n[5], 1.0f},
          {n[6], 1.0f},
          {n[7], 100.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{
              {n[0],
               ParallelSplit{
                   {n[6],
                    SeriesSplit{
                        {n[2],
                         ParallelSplit{
                             {n[7],
                              SeriesSplit{
                                  {n[1], ParallelSplit{{n[3], n[4]}}}}}}}}}},
               n[5]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("Transitive Edge") {
      DiGraph g2 = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> m = add_nodes(g2, 7);
      add_edges(g2,
                {DirectedEdge{m[0], m[1]},
                 DirectedEdge{m[0], m[2]},
                 DirectedEdge{m[1], m[3]},
                 DirectedEdge{m[1], m[4]},
                 DirectedEdge{m[2], m[5]},
                 DirectedEdge{m[3], m[6]},
                 DirectedEdge{m[4], m[5]},
                 DirectedEdge{m[5], m[6]}});

      std::unordered_map<Node, float> cost_map2 = {
          {m[0], 1.0f},
          {m[1], 1.0f},
          {m[2], 1.0f},
          {m[3], 10.0f},
          {m[4], 1.0f},
          {m[5], 10.0f},
          {m[6], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g2, cost_map2);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{m[0],
                       ParallelSplit{{SeriesSplit{{m[1], m[4]}}, m[2]}},
                       ParallelSplit{{m[3], m[5]}},
                       m[6]}}};

      CHECK(result == correct);
      CHECK(dependencies_are_maintained(g2, result));
    }

    SUBCASE("Graph From Paper - constant cost map") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 18);
      add_edges(g, {DirectedEdge{n[0], n[1]},   DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},   DirectedEdge{n[1], n[4]},
                    DirectedEdge{n[2], n[10]},  DirectedEdge{n[2], n[11]},
                    DirectedEdge{n[2], n[12]},  DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[3], n[6]},   DirectedEdge{n[4], n[6]},
                    DirectedEdge{n[4], n[7]},   DirectedEdge{n[4], n[10]},
                    DirectedEdge{n[5], n[8]},   DirectedEdge{n[6], n[8]},
                    DirectedEdge{n[6], n[9]},   DirectedEdge{n[7], n[8]},
                    DirectedEdge{n[8], n[17]},  DirectedEdge{n[9], n[17]},
                    DirectedEdge{n[10], n[9]},  DirectedEdge{n[10], n[16]},
                    DirectedEdge{n[11], n[16]}, DirectedEdge{n[12], n[13]},
                    DirectedEdge{n[12], n[14]}, DirectedEdge{n[13], n[15]},
                    DirectedEdge{n[14], n[15]}, DirectedEdge{n[15], n[16]},
                    DirectedEdge{n[16], n[17]}});

      std::unordered_map<Node, float> cost_map;
      for (int i = 0; i < 18; i++) {
        cost_map[n[i]] = 1.0f;
      }

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{
              {n[0],
               ParallelSplit{
                   {SeriesSplit{{n[1], ParallelSplit{{n[3], n[4]}}}},
                    SeriesSplit{{n[2], ParallelSplit{{n[11], n[12]}}}}}},
               ParallelSplit{
                   {SeriesSplit{{ParallelSplit{{n[10], n[5], n[6], n[7]}},
                                 ParallelSplit{{n[8], n[9]}}}},
                    SeriesSplit{{ParallelSplit{{n[13], n[14]}}, n[15]}}}},
               n[16],
               n[17]}}};

      CHECK(dependencies_are_maintained(g, result));
      CHECK(result == correct);
    }

    SUBCASE("Graph From Paper - non constant cost map") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 18);
      add_edges(g, {DirectedEdge{n[0], n[1]},   DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},   DirectedEdge{n[1], n[4]},
                    DirectedEdge{n[2], n[10]},  DirectedEdge{n[2], n[11]},
                    DirectedEdge{n[2], n[12]},  DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[3], n[6]},   DirectedEdge{n[4], n[6]},
                    DirectedEdge{n[4], n[7]},   DirectedEdge{n[4], n[10]},
                    DirectedEdge{n[5], n[8]},   DirectedEdge{n[6], n[8]},
                    DirectedEdge{n[6], n[9]},   DirectedEdge{n[7], n[8]},
                    DirectedEdge{n[8], n[17]},  DirectedEdge{n[9], n[17]},
                    DirectedEdge{n[10], n[16]}, DirectedEdge{n[11], n[16]},
                    DirectedEdge{n[12], n[13]}, DirectedEdge{n[12], n[14]},
                    DirectedEdge{n[13], n[15]}, DirectedEdge{n[14], n[15]},
                    DirectedEdge{n[15], n[16]}, DirectedEdge{n[16], n[17]}});

      std::unordered_map<Node, float> cost_map;
      for (int i = 0; i < 18; i++) {
        cost_map[n[i]] = 1.0f;
      }

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      CHECK(dependencies_are_maintained(g, result));
    }

    SUBCASE("Additional Test Case") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 18);
      add_edges(g, {DirectedEdge{n[0], n[1]},   DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},   DirectedEdge{n[1], n[4]},
                    DirectedEdge{n[2], n[10]},  DirectedEdge{n[2], n[11]},
                    DirectedEdge{n[2], n[12]},  DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[3], n[6]},   DirectedEdge{n[4], n[6]},
                    DirectedEdge{n[4], n[7]},   DirectedEdge{n[4], n[10]},
                    DirectedEdge{n[5], n[8]},   DirectedEdge{n[6], n[8]},
                    DirectedEdge{n[6], n[9]},   DirectedEdge{n[7], n[8]},
                    DirectedEdge{n[8], n[17]},  DirectedEdge{n[9], n[17]},
                    DirectedEdge{n[10], n[16]}, DirectedEdge{n[11], n[16]},
                    DirectedEdge{n[12], n[13]}, DirectedEdge{n[12], n[14]},
                    DirectedEdge{n[13], n[15]}, DirectedEdge{n[14], n[15]},
                    DirectedEdge{n[15], n[16]}, DirectedEdge{n[16], n[17]}});

      std::unordered_map<Node, float> cost_map = {
          {n[0], 1.0f},
          {n[1], 3.0f},
          {n[2], 5.0f},
          {n[3], 3.0f},
          {n[4], 5.0f},
          {n[5], 3.0f},
          {n[6], 5.0f},
          {n[7], 1.0f},
          {n[8], 5.0f},
          {n[9], 5.0f},
          {n[10], 3.0f},
          {n[11], 3.0f},
          {n[12], 1.0f},
          {n[13], 3.0f},
          {n[14], 3.0f},
          {n[15], 1.0f},
          {n[16], 1.0f},
          {n[17], 1.0f},
      };

      SeriesParallelDecomposition result = flexible_sync(g, cost_map);
      CHECK(dependencies_are_maintained(g, result));
    }
  }
}
