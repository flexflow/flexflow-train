#include "utils/graph/digraph/algorithms/get_bottlenecks.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_bottlenecks") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("single node") {
      std::vector<Node> n = add_nodes(g, 1);
      std::unordered_set<Node> expected = {n.at(0)};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("linear graph") {
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(
          g, {DirectedEdge{n.at(0), n.at(1)}, DirectedEdge{n.at(1), n.at(2)}});

      std::unordered_set<Node> expected = {n.at(0), n.at(1), n.at(2)};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("rhombus") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(2), n.at(3)}});

      std::unordered_set<Node> expected = {n.at(0), n.at(3)};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("two rhombuses in serial") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(2), n.at(3)},
                 DirectedEdge{n.at(3), n.at(4)},
                 DirectedEdge{n.at(3), n.at(5)},
                 DirectedEdge{n.at(4), n.at(5)}});

      std::unordered_set<Node> expected = {n.at(0), n.at(3), n.at(5)};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("middle bottleneck") {
      std::vector<Node> n = add_nodes(g, 5);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(2), n.at(3)},
                 DirectedEdge{n.at(2), n.at(4)}});

      std::unordered_set<Node> expected = {};
      CHECK(get_bottlenecks(g) == expected);
    }

    SUBCASE("single source, multiple sinks") {
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(
          g, {DirectedEdge{n.at(0), n.at(1)}, DirectedEdge{n.at(0), n.at(2)}});

      std::unordered_set<Node> expected = {n.at(0)};
      CHECK(get_bottlenecks(g) == expected);
    }
  }
}
