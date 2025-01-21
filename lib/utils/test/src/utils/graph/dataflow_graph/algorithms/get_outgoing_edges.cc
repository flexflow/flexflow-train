#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_outgoing_edges(DataflowGraphView, Node)") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({o1}, 1);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o1}, 1);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o2}, 1);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    SUBCASE("n2 - single outgoing edge") {
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, n2);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{o2, DataflowInput{n4, 0}},
      };
      CHECK(result == correct);
    }

    SUBCASE("n1 - multiple outgoing edges") {
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, n1);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{o1, DataflowInput{n2, 0}},
          DataflowEdge{o1, DataflowInput{n3, 0}},
      };
      CHECK(result == correct);
    }

    SUBCASE("n4 - no outgoing edges") {
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, n4);
      std::unordered_set<DataflowEdge> correct = {};
      CHECK(result == correct);
    }
  }

  TEST_CASE("get_outgoing_edges(DataflowGraphView, std::unordered_set<Node>)") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({o1}, 1);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o1}, 1);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o2}, 1);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    SUBCASE("multiple nodes - combined outgoing edges") {
      std::unordered_set<Node> nodes = {n1, n2};
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, nodes);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{o1, DataflowInput{n2, 0}},
          DataflowEdge{o1, DataflowInput{n3, 0}},
          DataflowEdge{o2, DataflowInput{n4, 0}},
      };
      CHECK(result == correct);
    }

    SUBCASE("multiple nodes - no outgoing edges") {
      std::unordered_set<Node> nodes = {n3, n4};
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, nodes);
      std::unordered_set<DataflowEdge> correct = {};
      CHECK(result == correct);
    }
  }
}
