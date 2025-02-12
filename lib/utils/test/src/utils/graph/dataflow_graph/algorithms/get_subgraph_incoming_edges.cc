#include "utils/graph/dataflow_graph/algorithms/get_subgraph_incoming_edges.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_subgraph_incoming_edges(DataflowGraphView, "
            "std::unordered_set<Node>") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1_n);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({o1}, 1_n);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o1, o2, o1}, 1_n);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o2, o3}, 1_n);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    std::unordered_set<Node> input_node_set = {n2, n3};

    std::unordered_set<DataflowEdge> result =
        get_subgraph_incoming_edges(g, input_node_set);

    std::unordered_set<DataflowEdge> correct = {
        DataflowEdge{o1, DataflowInput{n2, 0_n}},
        DataflowEdge{o1, DataflowInput{n3, 0_n}},
        DataflowEdge{o1, DataflowInput{n3, 2_n}},
    };

    CHECK(result == correct);
  }
}
