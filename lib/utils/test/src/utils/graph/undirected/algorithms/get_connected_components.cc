#include "utils/graph/undirected/algorithms/get_connected_components.h"
#include "utils/fmt/unordered_set.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/undirected/algorithms/make_undirected_edge.h"
#include "utils/graph/undirected/undirected_graph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_connected_components") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();

    SUBCASE("disjoint nodes") {
      std::vector<Node> n = add_nodes(g, 3);

      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0)},
          {n.at(1)},
          {n.at(2)},
      };
      std::unordered_set<std::unordered_set<Node>> result =
          get_connected_components(g);

      CHECK(correct == result);
    }

    SUBCASE("1 component") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    make_undirected_edge(n.at(0), n.at(1)),
                    make_undirected_edge(n.at(1), n.at(2)),
                    make_undirected_edge(n.at(2), n.at(3)),
                    make_undirected_edge(n.at(3), n.at(0)),
                });

      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0), n.at(1), n.at(2), n.at(3)},
      };
      std::unordered_set<std::unordered_set<Node>> result =
          get_connected_components(g);

      CHECK(correct == result);
    }

    SUBCASE("2 components") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    make_undirected_edge(n.at(0), n.at(1)),
                    make_undirected_edge(n.at(2), n.at(1)),
                });

      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0), n.at(1), n.at(2)},
          {n.at(3)},
      };
      std::unordered_set<std::unordered_set<Node>> result =
          get_connected_components(g);

      CHECK(correct == result);
    }

    SUBCASE("3 components") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    make_undirected_edge(n.at(0), n.at(1)),
                    make_undirected_edge(n.at(0), n.at(2)),
                    make_undirected_edge(n.at(1), n.at(2)),
                    make_undirected_edge(n.at(3), n.at(4)),
                });

      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0), n.at(1), n.at(2)},
          {n.at(3), n.at(4)},
          {n.at(5)},
      };
      std::unordered_set<std::unordered_set<Node>> result =
          get_connected_components(g);

      CHECK(correct == result);
    }

    SUBCASE("empty graph") {
      std::unordered_set<std::unordered_set<Node>> correct = {};
      std::unordered_set<std::unordered_set<Node>> result =
          get_connected_components(g);

      CHECK(correct == result);
    }
  }
}
