#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_longest_path_lengths_from_root") {
    SUBCASE("linear graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 5);
      std::vector<DirectedEdge> edges = {
          DirectedEdge{n.at(0), n.at(1)},
          DirectedEdge{n.at(1), n.at(2)},
          DirectedEdge{n.at(2), n.at(3)},
          DirectedEdge{n.at(3), n.at(4)},
      };

      add_edges(g, edges);

      std::unordered_map<Node, nonnegative_int> expected_lengths = {
          {n.at(0), 1_n},
          {n.at(1), 2_n},
          {n.at(2), 3_n},
          {n.at(3), 4_n},
          {n.at(4), 5_n},
      };

      CHECK(get_longest_path_lengths_from_root(g) == expected_lengths);
    }

    SUBCASE("more complex graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 7);
      std::vector<DirectedEdge> edges = {DirectedEdge{n.at(0), n.at(1)},
                                         DirectedEdge{n.at(0), n.at(3)},
                                         DirectedEdge{n.at(0), n.at(4)},
                                         DirectedEdge{n.at(0), n.at(6)},
                                         DirectedEdge{n.at(1), n.at(2)},
                                         DirectedEdge{n.at(2), n.at(3)},
                                         DirectedEdge{n.at(3), n.at(5)},
                                         DirectedEdge{n.at(4), n.at(5)},
                                         DirectedEdge{n.at(5), n.at(6)}};

      add_edges(g, edges);

      std::unordered_map<Node, nonnegative_int> expected_lengths = {
          {n.at(0), 1_n},
          {n.at(1), 2_n},
          {n.at(2), 3_n},
          {n.at(3), 4_n},
          {n.at(4), 2_n},
          {n.at(5), 5_n},
          {n.at(6), 6_n},
      };

      CHECK(get_longest_path_lengths_from_root(g) == expected_lengths);
    }
  }
}
