#include "utils/graph/digraph/algorithms.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_edges(DiGraphView)") {
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
      std::unordered_set<DirectedEdge> correct = unordered_set_of(e);
      std::unordered_set<DirectedEdge> result = get_edges(g);
      CHECK(result == correct);
    }

    SUBCASE("Adding an edge") {
      g.add_edge(DirectedEdge{n.at(3), n.at(1)});
      std::unordered_set<DirectedEdge> correct = {
          DirectedEdge{n.at(0), n.at(1)},
          DirectedEdge{n.at(0), n.at(2)},
          DirectedEdge{n.at(0), n.at(3)},
          DirectedEdge{n.at(1), n.at(2)},
          DirectedEdge{n.at(3), n.at(1)},
      };
      std::unordered_set<DirectedEdge> result = get_edges(g);
      CHECK(result == correct);
    }

    SUBCASE("Removing an edge") {
      g.remove_edge(DirectedEdge{n.at(0), n.at(3)});
      std::unordered_set<DirectedEdge> correct = {
          DirectedEdge{n.at(0), n.at(1)},
          DirectedEdge{n.at(0), n.at(2)},
          DirectedEdge{n.at(1), n.at(2)},
      };
      std::unordered_set<DirectedEdge> result = get_edges(g);
      CHECK(result == correct);
    }
  }

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

  TEST_CASE("get_initial_nodes(DiGraphView)") {
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
      std::unordered_set<Node> correct = {n.at(0)};
      std::unordered_set<Node> result = get_initial_nodes(g);
      CHECK(result == correct);
    }

    SUBCASE("Adding an edge to remove a source") {
      g.add_edge(DirectedEdge{n.at(2), n.at(0)});
      std::unordered_set<Node> correct = {};
      std::unordered_set<Node> result = get_initial_nodes(g);
      CHECK(result == correct);
    }

    SUBCASE("Removing an edge to create a new source") {
      g.remove_edge(DirectedEdge{n.at(0), n.at(1)});
      std::unordered_set<Node> correct = {n.at(0), n.at(1)};
      std::unordered_set<Node> result = get_initial_nodes(g);
      CHECK(result == correct);
    }

    SUBCASE("Creating a cycle") {
      g.add_edge(DirectedEdge{n.at(2), n.at(0)});
      std::unordered_set<Node> result = get_initial_nodes(g);
      std::unordered_set<Node> correct = {};
      CHECK(result.empty());
    }
  }
}
