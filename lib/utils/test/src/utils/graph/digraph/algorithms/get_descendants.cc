#include "utils/graph/digraph/algorithms/get_descendants.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_descendants") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("single node") {
      std::vector<Node> n = add_nodes(g, 1);

      std::unordered_set<Node> correct = {};
      std::unordered_set<Node> result = get_descendants(g, n.at(0));
      CHECK(correct == result);
    }

    SUBCASE("linear graph") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(2), n.at(3)}});

      SUBCASE("n.at(0)") {
        std::unordered_set<Node> correct = {n.at(1), n.at(2), n.at(3)};
        std::unordered_set<Node> result = get_descendants(g, n.at(0));
        CHECK(correct == result);
      }

      SUBCASE("n.at(1)") {
        std::unordered_set<Node> correct = {n.at(2), n.at(3)};
        std::unordered_set<Node> result = get_descendants(g, n.at(1));
        CHECK(correct == result);
      }

      SUBCASE("n.at(2)") {
        std::unordered_set<Node> correct = {n.at(3)};
        std::unordered_set<Node> result = get_descendants(g, n.at(2));
        CHECK(correct == result);
      }

      SUBCASE("n.at(3)") {
        std::unordered_set<Node> correct = {};
        std::unordered_set<Node> result = get_descendants(g, n.at(3));
        CHECK(correct == result);
      }
    }

    SUBCASE("rhombus") {
      std::vector<Node> n = add_nodes(g, 5);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                });

      SUBCASE("n.at(0)") {
        std::unordered_set<Node> correct = {n.at(1), n.at(2), n.at(3)};
        std::unordered_set<Node> result = get_descendants(g, n.at(0));
        CHECK(correct == result);
      }

      SUBCASE("n.at(1)") {
        std::unordered_set<Node> correct = {n.at(3)};
        std::unordered_set<Node> result = get_descendants(g, n.at(1));
        CHECK(correct == result);
      }

      SUBCASE("n.at(2)") {
        std::unordered_set<Node> correct = {n.at(3)};
        std::unordered_set<Node> result = get_descendants(g, n.at(2));
        CHECK(correct == result);
      }

      SUBCASE("n.at(3)") {
        std::unordered_set<Node> correct = {};
        std::unordered_set<Node> result = get_descendants(g, n.at(3));
        CHECK(correct == result);
      }
    }

    SUBCASE("disconnected graph") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(3), n.at(4)},
                });

      SUBCASE("n.at(0)") {
        std::unordered_set<Node> correct = {n.at(1), n.at(2)};
        std::unordered_set<Node> result = get_descendants(g, n.at(0));
        CHECK(correct == result);
      }

      SUBCASE("n.at(1)") {
        std::unordered_set<Node> correct = {n.at(2)};
        std::unordered_set<Node> result = get_descendants(g, n.at(1));
        CHECK(correct == result);
      }

      SUBCASE("n.at(2)") {
        std::unordered_set<Node> correct = {};
        std::unordered_set<Node> result = get_descendants(g, n.at(2));
        CHECK(correct == result);
      }

      SUBCASE("n.at(3)") {
        std::unordered_set<Node> correct = {n.at(4)};
        std::unordered_set<Node> result = get_descendants(g, n.at(3));
        CHECK(correct == result);
      }

      SUBCASE("n.at(4)") {
        std::unordered_set<Node> correct = {};
        std::unordered_set<Node> result = get_descendants(g, n.at(4));
        CHECK(correct == result);
      }
    }
  }
}
