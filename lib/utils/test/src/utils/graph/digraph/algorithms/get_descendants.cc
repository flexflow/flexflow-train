#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_descendants") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("single node") {
      std::vector<Node> nodes = add_nodes(g, 1);
      std::unordered_set<Node> expected = {};
      CHECK(get_descendants(g, nodes[0]) == expected);
    }

    SUBCASE("linear graph") {
      std::vector<Node> nodes = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{nodes[0], nodes[1]},
                 DirectedEdge{nodes[1], nodes[2]},
                 DirectedEdge{nodes[2], nodes[3]}});

      std::unordered_set<Node> expected_0 = {nodes[1], nodes[2], nodes[3]};
      std::unordered_set<Node> expected_1 = {nodes[2], nodes[3]};
      std::unordered_set<Node> expected_2 = {nodes[3]};
      std::unordered_set<Node> expected_3 = {};

      CHECK(get_descendants(g, nodes[0]) == expected_0);
      CHECK(get_descendants(g, nodes[1]) == expected_1);
      CHECK(get_descendants(g, nodes[2]) == expected_2);
      CHECK(get_descendants(g, nodes[3]) == expected_3);
    }

    SUBCASE("rhombus") {
      std::vector<Node> nodes = add_nodes(g, 5);
      add_edges(g,
                {
                    DirectedEdge{nodes[0], nodes[1]},
                    DirectedEdge{nodes[0], nodes[2]},
                    DirectedEdge{nodes[1], nodes[3]},
                    DirectedEdge{nodes[2], nodes[3]},
                });

      std::unordered_set<Node> expected_0 = {nodes[1], nodes[2], nodes[3]};
      std::unordered_set<Node> expected_1 = {nodes[3]};
      std::unordered_set<Node> expected_2 = {nodes[3]};
      std::unordered_set<Node> expected_3 = {};

      CHECK(get_descendants(g, nodes[0]) == expected_0);
      CHECK(get_descendants(g, nodes[1]) == expected_1);
      CHECK(get_descendants(g, nodes[2]) == expected_2);
      CHECK(get_descendants(g, nodes[3]) == expected_3);
    }

    SUBCASE("disconnected graph") {
      std::vector<Node> nodes = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{nodes[0], nodes[1]},
                    DirectedEdge{nodes[1], nodes[2]},
                    DirectedEdge{nodes[3], nodes[4]},
                });

      std::unordered_set<Node> expected_0 = {nodes[1], nodes[2]};
      std::unordered_set<Node> expected_1 = {nodes[2]};
      std::unordered_set<Node> expected_2 = {};
      std::unordered_set<Node> expected_3 = {nodes[4]};
      std::unordered_set<Node> expected_4 = {};

      CHECK(get_descendants(g, nodes[0]) == expected_0);
      CHECK(get_descendants(g, nodes[1]) == expected_1);
      CHECK(get_descendants(g, nodes[2]) == expected_2);
      CHECK(get_descendants(g, nodes[3]) == expected_3);
      CHECK(get_descendants(g, nodes[4]) == expected_4);
    }
  }
}
