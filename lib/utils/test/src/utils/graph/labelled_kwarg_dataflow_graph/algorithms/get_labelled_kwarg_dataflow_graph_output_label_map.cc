#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_output_label_map.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_labelled_kwarg_dataflow_graph_output_label_map") {
    LabelledKwargDataflowGraph<std::string, float, int> g =
        LabelledKwargDataflowGraph<std::string, float, int>::template create<
            UnorderedSetLabelledOpenKwargDataflowGraph<std::string,
                                                       float,
                                                       bool,
                                                       int>>();

    SUBCASE("graph is empty") {
      std::unordered_map<KwargDataflowOutput<int>, float> result =
          get_labelled_kwarg_dataflow_graph_output_label_map(
              static_cast<
                  LabelledKwargDataflowGraphView<std::string, float, int>>(g));

      std::unordered_map<KwargDataflowOutput<int>, float> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("graph is non-empty") {
      float n1_t1_label = 5.3;
      float n2_t1_label = 12.1;
      float n2_t2_label = 5.3;
      float n3_t1_label = 1.7;

      KwargNodeAddedResult<int> n1_added = g.add_node(
          /*node_label=*/"n1",
          /*inputs=*/{},
          /*output_labels=*/
          std::unordered_map<int, float>{
              {2, n1_t1_label},
          });
      Node n1 = n1_added.node;
      KwargDataflowOutput<int> n1_t1 = require_only_key(n1_added.outputs, 2);

      KwargNodeAddedResult<int> n2_added = g.add_node(
          /*node_label=*/"n2",
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
          /*node_label=*/"n1",
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

      std::unordered_map<KwargDataflowOutput<int>, float> result =
          get_labelled_kwarg_dataflow_graph_output_label_map(
              static_cast<
                  LabelledKwargDataflowGraphView<std::string, float, int>>(g));

      std::unordered_map<KwargDataflowOutput<int>, float> correct = {
          {n1_t1, n1_t1_label},
          {n2_t1, n2_t1_label},
          {n2_t2, n2_t2_label},
          {n3_t1, n3_t1_label},
      };

      CHECK(result == correct);
    }
  }
}
