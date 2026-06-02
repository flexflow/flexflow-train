#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_node_label_map.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_labelled_kwarg_dataflow_graph_node_label_map") {
    LabelledKwargDataflowGraph<std::string, float, int> g =
        LabelledKwargDataflowGraph<std::string, float, int>::template create<
            UnorderedSetLabelledOpenKwargDataflowGraph<std::string,
                                                       float,
                                                       bool,
                                                       int>>();

    SUBCASE("graph is empty") {
      std::unordered_map<Node, std::string> result =
          get_labelled_kwarg_dataflow_graph_node_label_map(
              static_cast<
                  LabelledKwargDataflowGraphView<std::string, float, int>>(g));

      std::unordered_map<Node, std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("graph is non-empty") {
      std::string n1_label = "n1";
      std::string n2_label = "n2";
      std::string n3_label = "n1";

      KwargNodeAddedResult<int> n1_added = g.add_node(
          /*node_label=*/n1_label,
          /*inputs=*/{},
          /*output_labels=*/
          std::unordered_map<int, float>{
              {2, 5.3},
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
              {0, 12.1},
              {1, 3.2},
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
              {4, 1.7},
          });
      Node n3 = n3_added.node;
      KwargDataflowOutput<int> n3_t1 = require_only_key(n3_added.outputs, 4);

      std::unordered_map<Node, std::string> result =
          get_labelled_kwarg_dataflow_graph_node_label_map(
              static_cast<
                  LabelledKwargDataflowGraphView<std::string, float, int>>(g));

      std::unordered_map<Node, std::string> correct = {
          {n1, n1_label},
          {n2, n2_label},
          {n3, n3_label},
      };

      CHECK(result == correct);
    }
  }
}
