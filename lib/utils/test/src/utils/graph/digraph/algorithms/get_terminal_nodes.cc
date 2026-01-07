#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_terminal_nodes(DiGraphView)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);
    std::vector<DirectedEdge> e = {
        DirectedEdge{n.at(0), n.at(1)},
        DirectedEdge{n.at(0), n.at(2)},
        DirectedEdge{n.at(0), n.at(3)},
        DirectedEdge{n.at(1), n.at(2)},
    };
    add_edges(g, e);

    SUBCASE("Base") {
      std::unordered_set<Node> correct = {n.at(2), n.at(3)};
      std::unordered_set<Node> result = get_terminal_nodes(g);
      CHECK(result == correct);
    }

    SUBCASE("Adding an edge to remove a terminal node") {
      g.add_edge(DirectedEdge{n.at(3), n.at(2)});
      std::unordered_set<Node> correct = {n.at(2)};
      std::unordered_set<Node> result = get_terminal_nodes(g);
      CHECK(result == correct);
    }

    SUBCASE("Creating a cycle") {
      g.add_edge(DirectedEdge{n.at(2), n.at(0)});
      std::unordered_set<Node> result = get_terminal_nodes(g);
      std::unordered_set<Node> correct = {n.at(3)};
      CHECK(result == correct);
    }
  }
}
