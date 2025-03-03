#include "utils/commutative_pair.h"
#include "utils/containers/repeat.h"
#include "utils/graph/instances/hashmap_undirected_graph.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/undirected/algorithms/make_undirected_edge.h"
#include "utils/graph/undirected/undirected_edge_query.h"
#include "utils/graph/undirected/undirected_graph.h"
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
      CHECK(g.query_nodes(node_query_all()) ==
            std::unordered_set<Node>{
                n.at(0), n.at(1), n.at(2), n.at(3), n.at(4)});

      CHECK(g.query_nodes(NodeQuery{query_set<Node>{{n.at(0), n.at(2)}}}) ==
            std::unordered_set<Node>{n.at(0), n.at(2)});
    }

    SUBCASE("query_edges") {

      std::unordered_set<UndirectedEdge> queried_edges =
          g.query_edges(undirected_edge_query_all());
      std::unordered_set<UndirectedEdge> expected = {
          e.at(0), e.at(1), e.at(2), e.at(3), e.at(4)};
      CHECK(queried_edges == expected);

      queried_edges = g.query_edges(
          UndirectedEdgeQuery{query_set<Node>{{n.at(0), n.at(1)}}});
      expected = std::unordered_set<UndirectedEdge>{e.at(0)};
      CHECK(queried_edges == expected);
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
