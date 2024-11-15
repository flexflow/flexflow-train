#include "utils/graph/digraph/algorithms/get_lowest_common_ancestors.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_lowest_common_ancestors") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("trees") {
      SUBCASE("single node") {
        std::vector<Node> n = add_nodes(g, 1);
        std::unordered_set<Node> correct = {n.at(0)};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(0)}).value();
        CHECK(correct == result);
      }

      SUBCASE("simple tree") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(
            g,
            {DirectedEdge{n.at(0), n.at(1)}, DirectedEdge{n.at(0), n.at(2)}});

        std::unordered_set<Node> correct = {n.at(0)};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(1), n.at(2)}).value();
        CHECK(correct == result);

        correct = {n.at(1)};
        result = get_lowest_common_ancestors(g, {n.at(1)}).value();
        CHECK(correct == result);

        correct = {n.at(2)};
        result = get_lowest_common_ancestors(g, {n.at(2)}).value();
        CHECK(correct == result);
      }

      SUBCASE("nodes at different heights") {
        std::vector<Node> n = add_nodes(g, 6);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(0), n.at(2)},
                   DirectedEdge{n.at(1), n.at(3)},
                   DirectedEdge{n.at(1), n.at(4)},
                   DirectedEdge{n.at(3), n.at(5)}});

        std::unordered_set<Node> correct = {n.at(0)};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(5), n.at(2)}).value();
        CHECK(correct == result);

        correct = {n.at(3)};
        result = get_lowest_common_ancestors(g, {n.at(5), n.at(3)}).value();
        CHECK(correct == result);

        correct = {n.at(1)};
        result = get_lowest_common_ancestors(g, {n.at(3), n.at(4)}).value();
        CHECK(correct == result);

        correct = {n.at(0)};
        result = get_lowest_common_ancestors(
                     g, {n.at(1), n.at(2), n.at(3), n.at(4), n.at(5)})
                     .value();
        CHECK(correct == result);
      }

      SUBCASE("straight path") {
        std::vector<Node> n = add_nodes(g, 4);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(1)},
                   DirectedEdge{n.at(1), n.at(2)},
                   DirectedEdge{n.at(2), n.at(3)}});

        std::unordered_set<Node> correct = {n.at(2)};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(2), n.at(3)}).value();
        CHECK(correct == result);

        correct = {n.at(1)};
        result = get_lowest_common_ancestors(g, {n.at(1), n.at(3)}).value();
        CHECK(correct == result);

        correct = {n.at(1)};
        result =
            get_lowest_common_ancestors(g, {n.at(1), n.at(2), n.at(3)}).value();
        CHECK(correct == result);
      }
    }

    SUBCASE("general dags") {

      SUBCASE("no LCA") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(
            g,
            {DirectedEdge{n.at(0), n.at(2)}, DirectedEdge{n.at(1), n.at(2)}});

        std::unordered_set<Node> correct = {};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(0), n.at(1)}).value();
        CHECK(correct == result);
      }

      SUBCASE("multiple LCAs") {
        std::vector<Node> n = add_nodes(g, 4);
        add_edges(g,
                  {DirectedEdge{n.at(0), n.at(2)},
                   DirectedEdge{n.at(1), n.at(2)},
                   DirectedEdge{n.at(0), n.at(3)},
                   DirectedEdge{n.at(1), n.at(3)}});

        std::unordered_set<Node> correct = {n.at(0), n.at(1)};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(2), n.at(3)}).value();
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

        std::unordered_set<Node> correct = {n.at(3)};
        std::unordered_set<Node> result =
            get_lowest_common_ancestors(g, {n.at(4), n.at(5)}).value();
        CHECK(correct == result);
      }
    }
  }
}
