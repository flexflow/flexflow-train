#include "utils/graph/digraph/algorithms/is_tree.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_tree") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("single node") {
      add_nodes(g, 1);
      CHECK(is_tree(g));
    }

    SUBCASE("simple tree") {
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                });
      CHECK(is_tree(g));
    }

    SUBCASE("simple cycle") {
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(0)},
                });
      CHECK_FALSE(is_tree(g));
    }

    SUBCASE("diamond pattern") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(2), n.at(3)}});
      CHECK_FALSE(is_tree(g));
    }

    SUBCASE("dowstream cycle") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(3), n.at(2)},
                });
      CHECK_FALSE(is_tree(g));
    }

    SUBCASE("multiple roots") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                });
      CHECK_FALSE(is_tree(g));
    }

    SUBCASE("multiple incoming edges") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(1), n.at(3)},
                });
      CHECK_FALSE(is_tree(g));
    }

    SUBCASE("crossing") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(3)},
                });
      CHECK_FALSE(is_tree(g));
    }
  }
}
