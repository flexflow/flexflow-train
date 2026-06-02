#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_data.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_labelled_kwarg_dataflow_graph_data") {
    LabelledKwargDataflowGraph<std::string, float, int> g =
        LabelledKwargDataflowGraph<std::string, float, int>::template create<
            UnorderedSetLabelledOpenKwargDataflowGraph<std::string,
                                                       float,
                                                       bool,
                                                       int>>();

    SUBCASE("graph is empty") {
      LabelledKwargDataflowGraphView<std::string, float, int> input = g;

      LabelledKwargDataflowGraphData<std::string, float, int> result =
          get_labelled_kwarg_dataflow_graph_data(input);

      LabelledKwargDataflowGraphData<std::string, float, int> correct =
          LabelledKwargDataflowGraphData<std::string, float, int>{
              /*node_data=*/std::unordered_map<Node, std::string>{},
              /*edges=*/std::unordered_set<KwargDataflowEdge<int>>{},
              /*output_data=*/
              std::unordered_map<KwargDataflowOutput<int>, float>{},
          };

      ASSERT(result == correct);
    }

    SUBCASE("graph is nonempty") {
      std::string n1_label = "n1";
      std::string n2_label = "n2";
      std::string n3_label = "n1";

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

      LabelledKwargDataflowGraphData<std::string, float, int> result =
          get_labelled_kwarg_dataflow_graph_data(input);

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

      LabelledKwargDataflowGraphData<std::string, float, int> correct =
          LabelledKwargDataflowGraphData<std::string, float, int>{
              /*node_data=*/std::unordered_map<Node, std::string>{
                  {n1, n1_label},
                  {n2, n2_label},
                  {n3, n3_label},
              },
              /*edges=*/
              std::unordered_set<KwargDataflowEdge<int>>{
                  mk_edge(n1, 2, n2, 3),
                  mk_edge(n1, 2, n3, 3),
                  mk_edge(n1, 2, n3, 1),
                  mk_edge(n2, 1, n3, 2),
              },
              /*output_data=*/
              std::unordered_map<KwargDataflowOutput<int>, float>{
                  {n1_t1, n1_t1_label},
                  {n2_t1, n2_t1_label},
                  {n2_t2, n2_t2_label},
                  {n3_t1, n3_t1_label},
              },
          };

      ASSERT(result == correct);
    }
  }
}
