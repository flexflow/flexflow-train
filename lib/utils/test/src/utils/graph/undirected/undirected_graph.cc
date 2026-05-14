#include "utils/graph/undirected/undirected_graph.h"
#include "utils/commutative_pair.h"
#include "utils/containers/repeat.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/undirected/algorithms/make_undirected_edge.h"
#include "utils/graph/undirected/undirected_edge_query.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE(
      "UndirectedGraph implementations", T, HashmapUndirectedGraph) {

    UndirectedGraph g = UndirectedGraph::create<T>();
    std::vector<Node> n = repeat(5_n, [&] { return g.add_node(); });
    std::vector<UndirectedEdge> e = {make_undirected_edge(n.at(0), n.at(1)),
                                     make_undirected_edge(n.at(0), n.at(2)),
                                     make_undirected_edge(n.at(1), n.at(2)),
                                     make_undirected_edge(n.at(2), n.at(4)),
                                     make_undirected_edge(n.at(1), n.at(3))};
    for (UndirectedEdge const &edge : e) {
      g.add_edge(edge);
    }

    SUBCASE("query_nodes") {
      SUBCASE("query_all") {
        std::unordered_set<Node> result = g.query_nodes(node_query_all());
        std::unordered_set<Node> correct = std::unordered_set<Node>{
            n.at(0),
            n.at(1),
            n.at(2),
            n.at(3),
            n.at(4),
        };

        CHECK(result == correct);
      }

      SUBCASE("query for specific nodes") {
        NodeQuery query = NodeQuery{
            query_set<Node>::match_values_in(std::set{n.at(0), n.at(2)}),
        };

        std::unordered_set<Node> result = g.query_nodes(query);
        std::unordered_set<Node> correct = std::unordered_set<Node>{
            n.at(0),
            n.at(2),
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("query_edges") {
      SUBCASE("query_all") {
        std::unordered_set<UndirectedEdge> result =
            g.query_edges(undirected_edge_query_all());

        std::unordered_set<UndirectedEdge> correct = {
            e.at(0),
            e.at(1),
            e.at(2),
            e.at(3),
            e.at(4),
        };

        CHECK(result == correct);
      }

      SUBCASE("query for specific edge") {
        UndirectedEdgeQuery query = UndirectedEdgeQuery{
            query_set<Node>::match_values_in(std::set{n.at(0), n.at(1)}),
        };

        std::unordered_set<UndirectedEdge> result = g.query_edges(query);
        std::unordered_set<UndirectedEdge> correct =
            std::unordered_set<UndirectedEdge>{
                e.at(0),
            };

        CHECK(result == correct);
      }
    }

    SUBCASE("remove_node_unsafe") {
      g.remove_node_unsafe(n.at(0));

      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{n.at(1), n.at(2), n.at(3), n.at(4)});

      // removing a node also removes its adjacent edges
      CHECK(g.query_edges(undirected_edge_query_all()) ==
            std::unordered_set<UndirectedEdge>{e.at(2), e.at(3), e.at(4)});

      g.remove_node_unsafe(n.at(1));

      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{n.at(2), n.at(3), n.at(4)});

      CHECK(g.query_edges(undirected_edge_query_all()) ==
            std::unordered_set<UndirectedEdge>{e.at(3)});
    }

    SUBCASE("remove_edge") {
      g.remove_edge(e.at(0));

      CHECK(g.query_edges(undirected_edge_query_all()) ==
            std::unordered_set<UndirectedEdge>{
                e.at(1), e.at(2), e.at(3), e.at(4)});
      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{
                n.at(0), n.at(1), n.at(2), n.at(3), n.at(4)});

      g.remove_edge(e.at(1));
      g.remove_edge(e.at(3));
      CHECK(g.query_edges(undirected_edge_query_all()) ==
            std::unordered_set<UndirectedEdge>{e.at(2), e.at(4)});
    }
  }
}
