#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/view_from_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_kwarg_dataflow_graph_data.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("kwarg_dataflow_graph_view_with_labelling") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};
    Node n5 = Node{5};

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

    KwargDataflowOutput n1_0 = KwargDataflowOutput{n1, 0};
    KwargDataflowOutput n2_3 = KwargDataflowOutput{n2, 3};
    KwargDataflowOutput n3_1 = KwargDataflowOutput{n3, 1};
    KwargDataflowOutput n5_0 = KwargDataflowOutput{n5, 0};

    KwargDataflowEdge<int> e_n1_0_n2_1 = mk_edge(n1, 0, n2, 1);
    KwargDataflowEdge<int> e_n1_0_n3_0 = mk_edge(n1, 0, n3, 0);
    KwargDataflowEdge<int> e_n2_3_n4_1 = mk_edge(n2, 3, n4, 1);
    KwargDataflowEdge<int> e_n3_1_n5_1 = mk_edge(n3, 1, n5, 1);

    KwargDataflowGraphData<int> g_data =
        KwargDataflowGraphData<int>{/*nodes=*/{n1, n2, n3, n4, n5},
                                    /*edges=*/
                                    {
                                        e_n1_0_n2_1,
                                        e_n1_0_n3_0,
                                        e_n2_3_n4_1,
                                        e_n3_1_n5_1,
                                    },
                                    /*outputs=*/
                                    {
                                        n1_0,
                                        n2_3,
                                        n3_1,
                                        n5_0,
                                    }};

    KwargDataflowGraphView<int> g = view_from_kwarg_dataflow_graph_data(g_data);

    float n1_label = 3.5;
    float n2_label = 1.2;
    float n3_label = 1.2;
    float n4_label = 7.8;
    float n5_label = 2.2;

    std::unordered_map<Node, float> node_labelling = {
        {n1, 3.5},
        {n2, 1.2},
        {n3, 1.2},
        {n4, 7.8},
        {n5, 2.2},
    };

    std::string n1_0_label = "a";
    std::string n2_3_label = "b";
    std::string n3_1_label = "c";
    std::string n5_0_label = "d";

    std::unordered_map<KwargDataflowOutput<int>, std::string> value_labelling =
        {
            {n1_0, n1_0_label},
            {n2_3, n2_3_label},
            {n3_1, n3_1_label},
            {n5_0, n5_0_label},
        };

    LabelledKwargDataflowGraphView<float, std::string, int> result =
        kwarg_dataflow_graph_view_with_labelling(
            g, node_labelling, value_labelling);
    LabelledKwargDataflowGraphData<float, std::string, int> result_data =
        get_labelled_kwarg_dataflow_graph_data(result);

    LabelledKwargDataflowGraphData<float, std::string, int> correct_data =
        LabelledKwargDataflowGraphData<float, std::string, int>{
            /*node_data=*/node_labelling,
            /*edges=*/
            {
                e_n1_0_n2_1,
                e_n1_0_n3_0,
                e_n2_3_n4_1,
                e_n3_1_n5_1,
            },
            /*output_data=*/value_labelling,
        };

    CHECK(result_data == correct_data);
  }
}
