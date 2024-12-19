#include "utils/graph/digraph/algorithms/get_bottlenecks.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_bottlenecks") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("single node") {
      std::vector<Node> nodes = add_nodes(g, 1);
      std::unordered_set<Node> expected = {nodes[0]};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("linear graph") {
      std::vector<Node> nodes = add_nodes(g, 3);
      add_edges(
          g,
          {DirectedEdge{nodes[0], nodes[1]}, DirectedEdge{nodes[1], nodes[2]}});

      std::unordered_set<Node> expected = {nodes[0], nodes[1], nodes[2]};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("rhombus") {
      std::vector<Node> nodes = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{nodes[0], nodes[1]},
                 DirectedEdge{nodes[0], nodes[2]},
                 DirectedEdge{nodes[1], nodes[3]},
                 DirectedEdge{nodes[2], nodes[3]}});

      std::unordered_set<Node> expected = {nodes[0], nodes[3]};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("two rhombuses in serial") {
      std::vector<Node> nodes = add_nodes(g, 6);
      add_edges(g,
                {DirectedEdge{nodes[0], nodes[1]},
                 DirectedEdge{nodes[0], nodes[2]},
                 DirectedEdge{nodes[1], nodes[3]},
                 DirectedEdge{nodes[2], nodes[3]},
                 DirectedEdge{nodes[3], nodes[4]},
                 DirectedEdge{nodes[3], nodes[5]},
                 DirectedEdge{nodes[4], nodes[5]}});

      std::unordered_set<Node> expected = {nodes[0], nodes[3], nodes[5]};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("middle bottleneck") {
      std::vector<Node> nodes = add_nodes(g, 5);
      add_edges(g,
                {DirectedEdge{nodes[0], nodes[2]},
                 DirectedEdge{nodes[1], nodes[2]},
                 DirectedEdge{nodes[2], nodes[3]},
                 DirectedEdge{nodes[2], nodes[4]}});

      std::unordered_set<Node> expected = {};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("single source, multiple sinks") {
      std::vector<Node> nodes = add_nodes(g, 3);
      add_edges(
          g,
          {DirectedEdge{nodes[0], nodes[1]}, DirectedEdge{nodes[0], nodes[2]}});

      std::unordered_set<Node> expected = {nodes[0]};
      CHECK(get_bottlenecks(g) == expected);
    }
  }
}
