#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/containers/contains.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_values.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_full_graph_values_to_subgraph_inputs(OpenDataflowGraphView, "
            "std::unordered_set<Node>) ") {
    OpenDataflowGraph graph =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    DataflowGraphInput i0 = graph.add_input();
    DataflowGraphInput i1 = graph.add_input();
    DataflowGraphInput i2 = graph.add_input();

    NodeAddedResult n0_added = graph.add_node({OpenDataflowValue{i0}}, 1_n);
    Node n0 = n0_added.node;
    OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

    NodeAddedResult n1_added = graph.add_node({v0, OpenDataflowValue{i1}}, 1_n);
    Node n1 = n1_added.node;
    OpenDataflowValue v1 = OpenDataflowValue{get_only(n1_added.outputs)};

    NodeAddedResult n2_added = graph.add_node({v0}, 1_n);
    Node n2 = n2_added.node;
    OpenDataflowValue v2 = OpenDataflowValue{get_only(n2_added.outputs)};

    NodeAddedResult n3_added =
        graph.add_node({OpenDataflowValue{i2}, v1, v2}, 1_n);
    Node n3 = n3_added.node;

    std::unordered_set<Node> subgraph_nodes = {n1, n2, n3};

    bidict<OpenDataflowValue, DataflowGraphInput>
        full_graph_values_to_subgraph_inputs =
            get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

    SUBCASE("left entries are correct") {
      std::unordered_set<OpenDataflowValue> correct = {
          v0, OpenDataflowValue{i1}, OpenDataflowValue{i2}};
      CHECK(left_entries(full_graph_values_to_subgraph_inputs) == correct);
    }

    SUBCASE("mapping is correct") {
      CHECK(full_graph_values_to_subgraph_inputs.at_l(OpenDataflowValue{i1}) ==
            i1);
      CHECK(full_graph_values_to_subgraph_inputs.at_l(OpenDataflowValue{i2}) ==
            i2);
      std::unordered_set<DataflowGraphInput> inputs = {i1, i2};
      CHECK(!contains(inputs, full_graph_values_to_subgraph_inputs.at_l(v0)));
    }
  }

  TEST_CASE(
      "get_subgraph_data(OpenDataflowGraphView, std::unordered_set<Node>, "
      "bidict<OpenDataflowValue, DataflowGraphInput>)") {
    SUBCASE("2-node graph without inputs") {
      OpenDataflowGraph graph =
          OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

      NodeAddedResult n0_added = graph.add_node({}, 1_n);
      Node n0 = n0_added.node;
      OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

      NodeAddedResult n1_added = graph.add_node({v0}, 1_n);
      Node n1 = n1_added.node;

      SUBCASE("subgraph is full graph") {
        std::unordered_set<Node> subgraph_nodes = {n0, n1};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);
        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {OpenDataflowEdge{
                DataflowEdge{DataflowOutput{n0, 0_n}, DataflowInput{n1, 0_n}}}},
            {},
            {
                DataflowOutput{
                    n0,
                    0_n,
                },
                DataflowOutput{
                    n1,
                    0_n,
                },
            }};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is n0") {
        std::unordered_set<Node> subgraph_nodes = {n0};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);
        OpenDataflowGraphData correct = OpenDataflowGraphData{subgraph_nodes,
                                                              {},
                                                              {},
                                                              {DataflowOutput{
                                                                  n0,
                                                                  0_n,
                                                              }}};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is n1") {
        std::unordered_set<Node> subgraph_nodes = {n1};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);

        DataflowGraphInput n0_as_subgraph_input =
            full_graph_values_to_subgraph_inputs.at_l(v0);

        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {OpenDataflowEdge{DataflowInputEdge{n0_as_subgraph_input,
                                                DataflowInput{n1, 0_n}}}},
            {n0_as_subgraph_input},
            {DataflowOutput{
                n1,
                0_n,
            }}};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is empty") {
        std::unordered_set<Node> subgraph_nodes = {};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);
        OpenDataflowGraphData correct =
            OpenDataflowGraphData{subgraph_nodes, {}, {}, {}};
        CHECK(result == correct);
      }
    }

    SUBCASE("3-node graph with inputs") {
      OpenDataflowGraph graph =
          OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

      DataflowGraphInput i0 = graph.add_input();
      DataflowGraphInput i1 = graph.add_input();

      NodeAddedResult n0_added = graph.add_node({OpenDataflowValue{i0}}, 1_n);
      Node n0 = n0_added.node;
      OpenDataflowValue v0 = OpenDataflowValue{get_only(n0_added.outputs)};

      NodeAddedResult n1_added =
          graph.add_node({v0, OpenDataflowValue{i1}}, 1_n);
      Node n1 = n1_added.node;

      NodeAddedResult n2_added = graph.add_node({v0}, 1_n);
      Node n2 = n2_added.node;

      SUBCASE("subgraph is full graph") {
        std::unordered_set<Node> subgraph_nodes = {n0, n1, n2};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);

        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {
                OpenDataflowEdge{DataflowInputEdge{i0, DataflowInput{n0, 0_n}}},
                OpenDataflowEdge{DataflowInputEdge{i1, DataflowInput{n1, 1_n}}},
                OpenDataflowEdge{DataflowEdge{DataflowOutput{n0, 0_n},
                                              DataflowInput{n1, 0_n}}},
                OpenDataflowEdge{{DataflowEdge{DataflowOutput{n0, 0_n},
                                               DataflowInput{n2, 0_n}}}},
            },
            {i0, i1},
            {
                DataflowOutput{
                    n0,
                    0_n,
                },
                DataflowOutput{
                    n1,
                    0_n,
                },
                DataflowOutput{
                    n2,
                    0_n,
                },
            }};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is (n0, n1) split") {
        std::unordered_set<Node> subgraph_nodes = {n0, n1};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);

        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {
                OpenDataflowEdge{DataflowInputEdge{i0, DataflowInput{n0, 0_n}}},
                OpenDataflowEdge{DataflowInputEdge{i1, DataflowInput{n1, 1_n}}},
                OpenDataflowEdge{DataflowEdge{DataflowOutput{n0, 0_n},
                                              DataflowInput{n1, 0_n}}},
            },
            {i0, i1},
            {
                DataflowOutput{
                    n0,
                    0_n,
                },
                DataflowOutput{
                    n1,
                    0_n,
                },
            }};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is (n0, n1) split") {
        std::unordered_set<Node> subgraph_nodes = {n0, n1};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);

        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {
                OpenDataflowEdge{DataflowInputEdge{i0, DataflowInput{n0, 0_n}}},
                OpenDataflowEdge{DataflowInputEdge{i1, DataflowInput{n1, 1_n}}},
                OpenDataflowEdge{DataflowEdge{DataflowOutput{n0, 0_n},
                                              DataflowInput{n1, 0_n}}},
            },
            {i0, i1},
            {
                DataflowOutput{
                    n0,
                    0_n,
                },
                DataflowOutput{
                    n1,
                    0_n,
                },
            }};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is (n0, n2) split") {
        std::unordered_set<Node> subgraph_nodes = {n0, n2};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);

        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {
                OpenDataflowEdge{DataflowInputEdge{i0, DataflowInput{n0, 0_n}}},
                OpenDataflowEdge{DataflowEdge{DataflowOutput{n0, 0_n},
                                              DataflowInput{n2, 0_n}}},
            },
            {i0},
            {
                DataflowOutput{
                    n0,
                    0_n,
                },
                DataflowOutput{
                    n2,
                    0_n,
                },
            }};
        CHECK(result == correct);
      }

      SUBCASE("subgraph is (n1, n2) split") {
        std::unordered_set<Node> subgraph_nodes = {n1, n2};

        bidict<OpenDataflowValue, DataflowGraphInput>
            full_graph_values_to_subgraph_inputs =
                get_full_graph_values_to_subgraph_inputs(graph, subgraph_nodes);

        OpenDataflowGraphData result = get_subgraph_data(
            graph, subgraph_nodes, full_graph_values_to_subgraph_inputs);

        DataflowGraphInput n0_as_subgraph_input =
            full_graph_values_to_subgraph_inputs.at_l(OpenDataflowValue{v0});

        OpenDataflowGraphData correct = OpenDataflowGraphData{
            subgraph_nodes,
            {
                OpenDataflowEdge{DataflowInputEdge{i1, DataflowInput{n1, 1_n}}},
                OpenDataflowEdge{DataflowInputEdge{n0_as_subgraph_input,
                                                   DataflowInput{n1, 0_n}}},
                OpenDataflowEdge{DataflowInputEdge{n0_as_subgraph_input,
                                                   DataflowInput{n2, 0_n}}},
            },
            {i1, n0_as_subgraph_input},
            {
                DataflowOutput{
                    n1,
                    0_n,
                },
                DataflowOutput{
                    n2,
                    0_n,
                },
            }};
        CHECK(result == correct);
      }
    }
  }
}
