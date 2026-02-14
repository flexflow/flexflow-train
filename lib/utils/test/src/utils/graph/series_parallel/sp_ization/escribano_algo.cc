#include "utils/graph/series_parallel/sp_ization/escribano_algo.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
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
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("escribano_algo - subcomponents") {
    SUBCASE("add_dummy_nodes") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                });
      std::unordered_map<Node, NodeRole> node_types = {
          {n.at(0), NodeRole::PURE},
          {n.at(1), NodeRole::PURE},
          {n.at(2), NodeRole::PURE},
          {n.at(3), NodeRole::PURE},
      };

      DiGraph result = add_dummy_nodes(g, node_types);
      CHECK(get_edges(result).size() == 6);
      CHECK(get_nodes(result).size() == 6);
      CHECK(get_incoming_edges(g, n.at(3)).size() == 2);
      CHECK(get_outgoing_edges(g, n.at(0)).size() == 2);

      CHECK(node_types.size() == 6);
      CHECK(values(node_types) ==
            std::unordered_multiset<NodeRole>{NodeRole::PURE,
                                              NodeRole::PURE,
                                              NodeRole::PURE,
                                              NodeRole::PURE,
                                              NodeRole::DUMMY,
                                              NodeRole::DUMMY});
    }

    SUBCASE("get_component") {
      SUBCASE("2 layer graph, single simple component") {
        DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 4);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(1), n.at(2)},
                   DirectedEdge{n.at(1), n.at(3)}});
        std::unordered_map<Node, NodeRole> node_roles = {
            {n.at(0), NodeRole::PURE},
            {n.at(1), NodeRole::SYNC},
            {n.at(2), NodeRole::PURE},
            {n.at(3), NodeRole::PURE},
        };
        std::unordered_map<Node, nonnegative_int> depth_map = {
            {n.at(0), 0_n},
            {n.at(2), 1_n},
            {n.at(3), 1_n},
        };
        std::unordered_set<Node> correct = {n.at(0), n.at(2), n.at(3)};
        std::unordered_set<Node> result =
            get_component(g, n.at(2), depth_map, node_roles);
        CHECK(correct == result);
      }
      SUBCASE("2 layer graph, single complex component") {
        DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(2)},
                   DirectedEdge{n.at(1), n.at(3)},
                   DirectedEdge{n.at(2), n.at(4)},
                   DirectedEdge{n.at(3), n.at(4)},
                   DirectedEdge{n.at(3), n.at(5)}});
        std::unordered_map<Node, NodeRole> node_roles = {
            {n.at(0), NodeRole::PURE},
            {n.at(1), NodeRole::PURE},
            {n.at(2), NodeRole::SYNC},
            {n.at(3), NodeRole::SYNC},
            {n.at(4), NodeRole::PURE},
            {n.at(5), NodeRole::PURE},
        };
        std::unordered_map<Node, nonnegative_int> depth_map = {
            {n.at(0), 0_n},
            {n.at(1), 0_n},
            {n.at(4), 1_n},
            {n.at(5), 1_n},
        };
        SUBCASE("n.at(4)'s component") {
          std::unordered_set<Node> correct = {
              n.at(0), n.at(1), n.at(4), n.at(5)};
          std::unordered_set<Node> result =
              get_component(g, n.at(4), depth_map, node_roles);
          CHECK(correct == result);
        }
        SUBCASE("n.at(5)'s component") {
          std::unordered_set<Node> correct = {
              n.at(0), n.at(1), n.at(4), n.at(5)};
          std::unordered_set<Node> result =
              get_component(g, n.at(5), depth_map, node_roles);
          CHECK(correct == result);
        }
      }
      SUBCASE("3 layer graph, single connected component") {
        DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 7);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(1), n.at(2)},
                   DirectedEdge{n.at(1), n.at(3)},
                   DirectedEdge{n.at(2), n.at(4)},
                   DirectedEdge{n.at(3), n.at(4)},
                   DirectedEdge{n.at(4), n.at(5)},
                   DirectedEdge{n.at(4), n.at(6)}});
        std::unordered_map<Node, NodeRole> node_roles = {
            {n.at(0), NodeRole::PURE},
            {n.at(1), NodeRole::SYNC},
            {n.at(2), NodeRole::PURE},
            {n.at(3), NodeRole::PURE},
            {n.at(4), NodeRole::SYNC},
            {n.at(5), NodeRole::PURE},
            {n.at(6), NodeRole::PURE}};

        std::unordered_map<Node, nonnegative_int> depth_map = {{n.at(0), 0_n},
                                                   {n.at(2), 1_n},
                                                   {n.at(3), 1_n},
                                                   {n.at(5), 2_n},
                                                   {n.at(6), 2_n}};
        SUBCASE("n.at(5)'s component") {
          std::unordered_set<Node> correct = {
              n.at(2), n.at(3), n.at(5), n.at(6)};
          std::unordered_set<Node> result =
              get_component(g, n.at(5), depth_map, node_roles);
          CHECK(correct == result);
        }

        SUBCASE("n.at(6)'s component") {
          std::unordered_set<Node> correct = {
              n.at(2), n.at(3), n.at(5), n.at(6)};
          std::unordered_set<Node> result =
              get_component(g, n.at(6), depth_map, node_roles);
          CHECK(correct == result);
        }
      }
      SUBCASE("3 layer graph, multiple weakly connected components") {
        DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 10);
        add_edges(g,
                  {
                      DirectedEdge{n.at(0), n.at(1)},
                      DirectedEdge{n.at(1), n.at(2)},
                      DirectedEdge{n.at(1), n.at(3)},
                      DirectedEdge{n.at(1), n.at(4)},
                      DirectedEdge{n.at(2), n.at(5)},
                      DirectedEdge{n.at(3), n.at(6)},
                      DirectedEdge{n.at(4), n.at(6)},
                      DirectedEdge{n.at(5), n.at(7)},
                      DirectedEdge{n.at(5), n.at(8)},
                      DirectedEdge{n.at(6), n.at(9)},
                  });
        std::unordered_map<Node, NodeRole> node_roles = {
            {n.at(0), NodeRole::PURE},
            {n.at(1), NodeRole::SYNC},
            {n.at(2), NodeRole::PURE},
            {n.at(3), NodeRole::PURE},
            {n.at(4), NodeRole::PURE},
            {n.at(5), NodeRole::SYNC},
            {n.at(6), NodeRole::SYNC},
            {n.at(7), NodeRole::PURE},
            {n.at(8), NodeRole::PURE},
            {n.at(9), NodeRole::PURE},
        };

        std::unordered_map<Node, nonnegative_int> depth_map = {{n.at(0), 0_n},
                                                   {n.at(2), 1_n},
                                                   {n.at(3), 1_n},
                                                   {n.at(4), 1_n},
                                                   {n.at(7), 2_n},
                                                   {n.at(8), 2_n},
                                                   {n.at(9), 2_n}};
        SUBCASE("n.at(7)'s component") {
          std::unordered_set<Node> correct = {n.at(2), n.at(7), n.at(8)};
          std::unordered_set<Node> result =
              get_component(g, n.at(7), depth_map, node_roles);
          CHECK(correct == result);
        }
        SUBCASE("n.at(8)'s component") {
          std::unordered_set<Node> correct = {n.at(2), n.at(7), n.at(8)};
          std::unordered_set<Node> result =
              get_component(g, n.at(8), depth_map, node_roles);
          CHECK(correct == result);
        }
        SUBCASE("n.at(9)'s component") {
          std::unordered_set<Node> correct = {n.at(3), n.at(4), n.at(9)};
          std::unordered_set<Node> result =
              get_component(g, n.at(9), depth_map, node_roles);
          CHECK(correct == result);
        }
      }
    }
  }

  TEST_CASE("escribano_algorithm") {

    SUBCASE("Single Node") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      Node n = g.add_node();
      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{Node{n}};
      CHECK(sp == correct);
      CHECK(dependencies_are_maintained(g, sp));
    }
    SUBCASE("Linear Graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(g, {DirectedEdge{n[0], n[1]}, DirectedEdge{n[1], n[2]}});
      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      CHECK(dependencies_are_maintained(g, sp));
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n[0], n[1], n[2]}}};
      CHECK(sp == correct);
    }

    SUBCASE("Rhombus") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
                 DirectedEdge{n[0], n[2]},
                 DirectedEdge{n[1], n[3]},
                 DirectedEdge{n[2], n[3]}});
      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0], ParallelSplit{{n[1], n[2]}}, n[3]}}};

      CHECK(dependencies_are_maintained(g, sp));
      CHECK(correct == sp);
    }

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
      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      CHECK(dependencies_are_maintained(g, sp));
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0], n[1], ParallelSplit{{n[2], n[3]}}, n[4], n[5]}}};
      CHECK(sp == correct);
    }

    SUBCASE("Diamond without crossing") {
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

      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      CHECK(dependencies_are_maintained(g, sp));
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       ParallelSplit{{SeriesSplit{{n[1], n[3], n[4]}}, n[2]}},
                       n[5]}}};
      SeriesParallelDecomposition result = sp;
      CHECK(correct == result);
    }

    SUBCASE("Diamond Graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[4]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[4]},
                    DirectedEdge{n[3], n[5]},
                    DirectedEdge{n[4], n[5]},
                });
      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      CHECK(dependencies_are_maintained(g, sp));
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n[0],
                                                   ParallelSplit{{n[1], n[2]}},
                                                   ParallelSplit{{n[3], n[4]}},
                                                   n[5]}}};
      CHECK(sp == correct);
    }

    SUBCASE("Sample Graph #2") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 10);
      add_edges(g,
                {DirectedEdge{n[0], n[1]},
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
                 DirectedEdge{n[6], n[9]},
                 DirectedEdge{n[8], n[9]}});
      SeriesParallelDecomposition sp = escribano_sp_ization(g);
      CHECK(dependencies_are_maintained(g, sp));

      SeriesParallelDecomposition correct = SeriesParallelDecomposition{
          SeriesSplit{{n[0],
                       ParallelSplit{{n[1], n[3]}},
                       ParallelSplit{
                           {SeriesSplit{{n[2], n[6]}},
                            SeriesSplit{{ParallelSplit{
                                             {SeriesSplit{{n[5], n[7]}}, n[4]}},
                                         n[8]}}}},
                       n[9]}}};
      CHECK(sp == correct);
    }
  }
}

