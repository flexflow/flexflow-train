#include "utils/graph/digraph/algorithms/get_lowest_common_ancestors.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_lowest_common_ancestors") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("returns nullopt for empty input") {
      SUBCASE("empty graph") {
        std::optional<std::unordered_set<Node>> correct = std::nullopt;
        std::optional<std::unordered_set<Node>> result =
            get_lowest_common_ancestors(g, {});
        CHECK(correct == result);
      }

      SUBCASE("non-empty graph with empty set") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(
            g,
            {DirectedEdge{n.at(0), n.at(1)}, DirectedEdge{n.at(0), n.at(2)}});
        std::optional<std::unordered_set<Node>> correct = std::nullopt;
        std::optional<std::unordered_set<Node>> result =
            get_lowest_common_ancestors(g, {});
        CHECK(correct == result);
      }
    }

    SUBCASE("trees") {
      SUBCASE("single node") {
        std::vector<Node> n = add_nodes(g, 1);
        std::optional<std::unordered_set<Node>> correct = 
            std::unordered_set<Node>{n.at(0)};
        std::optional<std::unordered_set<Node>> result =
            get_lowest_common_ancestors(g, {n.at(0)});
        CHECK(correct == result);
      }

      SUBCASE("simple tree") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(
            g,
            {DirectedEdge{n.at(0), n.at(1)}, DirectedEdge{n.at(0), n.at(2)}});

        SUBCASE("LCA of siblings is parent") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(0)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(1), n.at(2)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of a single node is itself") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(1)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(1)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of another single node is itself") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(2)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(2)});
          CHECK(correct == result);
        }
      }

      SUBCASE("nodes at different heights") {
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(0), n.at(2)},
                   DirectedEdge{n.at(1), n.at(3)},
                   DirectedEdge{n.at(1), n.at(4)},
                   DirectedEdge{n.at(3), n.at(5)}});

        SUBCASE("LCA of nodes at different depths (root is LCA)") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(0)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(5), n.at(2)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of node and its ancestor is the ancestor") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(3)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(5), n.at(3)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of siblings at depth 2") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(1)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(3), n.at(4)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of multiple nodes across different branches") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(0)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(
                  g, {n.at(1), n.at(2), n.at(3), n.at(4), n.at(5)});
          CHECK(correct == result);
        }
      }

      SUBCASE("straight path") {
        std::vector<Node> n = add_nodes(g, 4);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(1), n.at(2)},
                   DirectedEdge{n.at(2), n.at(3)}});

        SUBCASE("LCA of adjacent nodes in a path") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(2)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(2), n.at(3)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of non-adjacent nodes in a path") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(1)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(1), n.at(3)});
          CHECK(correct == result);
        }

        SUBCASE("LCA of multiple nodes in a path") {
          std::optional<std::unordered_set<Node>> correct =
              std::unordered_set<Node>{n.at(1)};
          std::optional<std::unordered_set<Node>> result =
              get_lowest_common_ancestors(g, {n.at(1), n.at(2), n.at(3)});
          CHECK(correct == result);
        }
      }
    }

    SUBCASE("general dags") {

      SUBCASE("no LCA") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(
            g,
            {DirectedEdge{n.at(0), n.at(2)}, DirectedEdge{n.at(1), n.at(2)}});

        std::optional<std::unordered_set<Node>> correct = 
            std::unordered_set<Node>{};
        std::optional<std::unordered_set<Node>> result =
            get_lowest_common_ancestors(g, {n.at(0), n.at(1)});
        CHECK(correct == result);
      }

      SUBCASE("multiple LCAs") {
        std::vector<Node> n = add_nodes(g, 4);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(2)},
                   DirectedEdge{n.at(1), n.at(2)},
                   DirectedEdge{n.at(0), n.at(3)},
                   DirectedEdge{n.at(1), n.at(3)}});

        std::optional<std::unordered_set<Node>> correct = 
            std::unordered_set<Node>{n.at(0), n.at(1)};
        std::optional<std::unordered_set<Node>> result =
            get_lowest_common_ancestors(g, {n.at(2), n.at(3)});
        CHECK(correct == result);
      }

      SUBCASE("single LCA") {
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(0), n.at(2)},
                   DirectedEdge{n.at(2), n.at(3)},
                   DirectedEdge{n.at(1), n.at(4)},
                   DirectedEdge{n.at(3), n.at(4)},
                   DirectedEdge{n.at(3), n.at(5)},
                   DirectedEdge{n.at(1), n.at(5)}});

        std::optional<std::unordered_set<Node>> correct = 
            std::unordered_set<Node>{n.at(3)};
        std::optional<std::unordered_set<Node>> result =
            get_lowest_common_ancestors(g, {n.at(4), n.at(5)});
        CHECK(correct == result);
      }
    }
  }
}
