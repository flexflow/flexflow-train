#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/digraph/directed_edge_query.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("directed_edge_query_all") {
    Node n1{0}, n2{1}, n3{2};
    DirectedEdge e1 = DirectedEdge{n1, n2};
    DirectedEdge e2 = DirectedEdge{n2, n3};

    DirectedEdgeQuery result = directed_edge_query_all();

    CHECK(matches_edge(result, e1));
    CHECK(matches_edge(result, e2));
  }

  TEST_CASE("matches_edge") {
    Node n1{0}, n2{1}, n3{2};
    DirectedEdge e1 = DirectedEdge{n1, n2};
    DirectedEdge e2 = DirectedEdge{n2, n3};

    DirectedEdgeQuery query = DirectedEdgeQuery{query_set{n1}, query_set{n2}};

    CHECK(matches_edge(query, e1));
    CHECK_FALSE(matches_edge(query, e2));

    DirectedEdge flipped_edge = DirectedEdge{n2, n1};
    CHECK_FALSE(matches_edge(query, flipped_edge));
  }

  TEST_CASE("query_intersection") {
    Node n1{0}, n2{1}, n3{2}, n4{3};
    DirectedEdge e1 = DirectedEdge{n1, n2};
    DirectedEdge e2 = DirectedEdge{n2, n3};
    DirectedEdge e3 = DirectedEdge{n3, n4};

    SUBCASE("standard intersection") {
      DirectedEdgeQuery q1 =
          DirectedEdgeQuery{query_set{n1, n2}, query_set{n2, n3}};
      DirectedEdgeQuery q2 =
          DirectedEdgeQuery{query_set{n2, n3}, query_set{n3, n4}};

      DirectedEdgeQuery result = query_intersection(q1, q2);
      DirectedEdgeQuery expected =
          DirectedEdgeQuery{query_set{n2}, query_set{n3}};

      CHECK(result == expected);
    }

    SUBCASE("intersection with matchall") {
      DirectedEdgeQuery q1 =
          DirectedEdgeQuery{query_set{n1, n2}, matchall<Node>()};
      DirectedEdgeQuery q2 =
          DirectedEdgeQuery{matchall<Node>(), query_set{n3, n4}};

      DirectedEdgeQuery result = query_intersection(q1, q2);
      DirectedEdgeQuery expected =
          DirectedEdgeQuery{query_set{n1, n2}, query_set{n3, n4}};

      CHECK(result == expected);
    }
  }
}
