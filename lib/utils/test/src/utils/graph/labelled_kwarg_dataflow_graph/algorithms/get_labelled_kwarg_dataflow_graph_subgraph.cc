#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_subgraph.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_labelled_kwarg_dataflow_graph_subgraph") {
    LabelledKwargDataflowGraph<std::string, float, int> g =
        LabelledKwargDataflowGraph<std::string, float, int>::template create<
            UnorderedSetLabelledOpenKwargDataflowGraph<std::string,
                                                       float,
                                                       bool,
                                                       int>>();

    std::string n1_label = "apple";
    std::string n2_label = "banana";
    std::string n3_label = "apple";

    float n1_t1_label = 5.3;
    float n2_t1_label = 12.1;
    float n2_t2_label = 5.3;
    float n3_t1_label = 1.7;

    KwargNodeAddedResult<int> n1_added = g.add_node(
        /*node_label=*/n1_label,
        /*inputs=*/{},
        /*output_labels=*/
        std::unordered_map<int, float>{
            {2, n1_t1_label},
        });
    Node n1 = n1_added.node;
    KwargDataflowOutput<int> n1_t1 = require_only_key(n1_added.outputs, 2);

    KwargNodeAddedResult<int> n2_added = g.add_node(
        /*node_label=*/n2_label,
        /*inputs=*/
        std::unordered_map<int, KwargDataflowOutput<int>>{
            {3, n1_t1},
        },
        /*output_labels=*/
        std::unordered_map<int, float>{
            {0, n2_t1_label},
            {1, n2_t2_label},
        });
    Node n2 = n2_added.node;
    KwargDataflowOutput<int> n2_t1 = n2_added.outputs.at(0);
    KwargDataflowOutput<int> n2_t2 = n2_added.outputs.at(1);

    KwargNodeAddedResult<int> n3_added = g.add_node(
        /*node_label=*/n3_label,
        /*inputs=*/
        std::unordered_map<int, KwargDataflowOutput<int>>{
            {3, n1_t1},
            {1, n1_t1},
            {2, n2_t2},
        },
        /*output_labels=*/
        std::unordered_map<int, float>{
            {4, n3_t1_label},
        });
    Node n3 = n3_added.node;
    KwargDataflowOutput<int> n3_t1 = require_only_key(n3_added.outputs, 4);

    LabelledKwargDataflowGraphView<std::string, float, int> input = g;

    auto mk_edge = [](Node src,
                      int src_slot,
                      Node dst,
                      int dst_slot) -> KwargDataflowEdge<int> {
      return KwargDataflowEdge<int>{
          /*src=*/KwargDataflowOutput<int>{
              /*node=*/src,
              /*slot_name=*/src_slot,
          },
          /*dst=*/
          KwargDataflowInput<int>{
              /*node=*/dst,
              /*slot_name=*/dst_slot,
          },
      };
    };

    SUBCASE("node set includes all graph nodes") {
      LabelledKwargDataflowGraphView<std::string, float, int> result =
          get_labelled_kwarg_dataflow_graph_subgraph(
              input, std::unordered_set<Node>{n1, n2, n3});
      LabelledKwargDataflowGraphData<std::string, float, int> result_data =
          get_labelled_kwarg_dataflow_graph_data(result);

      LabelledKwargDataflowGraphData<std::string, float, int> correct_data =
          LabelledKwargDataflowGraphData<std::string, float, int>{
              /*node_data=*/{
                  {n1, n1_label},
                  {n2, n2_label},
                  {n3, n3_label},
              },
              /*edges=*/
              {
                  mk_edge(n1, 2, n2, 3),
                  mk_edge(n1, 2, n3, 3),
                  mk_edge(n1, 2, n3, 1),
                  mk_edge(n2, 1, n3, 2),
              },
              /*output_data=*/
              {
                  {n1_t1, n1_t1_label},
                  {n2_t1, n2_t1_label},
                  {n2_t2, n2_t2_label},
                  {n3_t1, n3_t1_label},
              },
          };

      CHECK(result_data == correct_data);
    }

    SUBCASE("node set includes only some graph nodes") {
      LabelledKwargDataflowGraphView<std::string, float, int> result =
          get_labelled_kwarg_dataflow_graph_subgraph(
              input, std::unordered_set<Node>{n2, n3});
      LabelledKwargDataflowGraphData<std::string, float, int> result_data =
          get_labelled_kwarg_dataflow_graph_data(result);

      LabelledKwargDataflowGraphData<std::string, float, int> correct_data =
          LabelledKwargDataflowGraphData<std::string, float, int>{
              /*node_data=*/{
                  {n2, n2_label},
                  {n3, n3_label},
              },
              /*edges=*/
              {
                  mk_edge(n2, 1, n3, 2),
              },
              /*output_data=*/
              {
                  {n2_t1, n2_t1_label},
                  {n2_t2, n2_t2_label},
                  {n3_t1, n3_t1_label},
              },
          };

      CHECK(result_data == correct_data);
    }

    SUBCASE("node set includes no graph nodes") {
      LabelledKwargDataflowGraphView<std::string, float, int> result =
          get_labelled_kwarg_dataflow_graph_subgraph(
              input, std::unordered_set<Node>{});
      LabelledKwargDataflowGraphData<std::string, float, int> result_data =
          get_labelled_kwarg_dataflow_graph_data(result);

      LabelledKwargDataflowGraphData<std::string, float, int> correct_data =
          LabelledKwargDataflowGraphData<std::string, float, int>{
              /*node_data=*/std::unordered_map<Node, std::string>{},
              /*edges=*/std::unordered_set<KwargDataflowEdge<int>>{},
              /*output_data=*/
              std::unordered_map<KwargDataflowOutput<int>, float>{},
          };

      CHECK(result_data == correct_data);
    }

    SUBCASE("node set includes nodes not in graph") {
      LabelledKwargDataflowGraphView<std::string, float, int>
          with_invalid_node = get_labelled_kwarg_dataflow_graph_subgraph(
              input, std::unordered_set<Node>{n2, n3, Node{100}});
      LabelledKwargDataflowGraphData<std::string, float, int>
          with_invalid_node_data =
              get_labelled_kwarg_dataflow_graph_data(with_invalid_node);

      LabelledKwargDataflowGraphView<std::string, float, int>
          without_invalid_node = get_labelled_kwarg_dataflow_graph_subgraph(
              input, std::unordered_set<Node>{n2, n3});
      LabelledKwargDataflowGraphData<std::string, float, int>
          without_invalid_node_data =
              get_labelled_kwarg_dataflow_graph_data(without_invalid_node);

      CHECK(with_invalid_node_data == without_invalid_node_data);
    }
  }
}
