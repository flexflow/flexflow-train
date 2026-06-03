#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_graph_input.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("view_from_open_kwarg_dataflow_graph_data") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};
    Node n5 = Node{5};

    auto mk_input_edge =
        [](KwargDataflowGraphInput<std::string> const &src,
           Node dst,
           int dst_slot) -> OpenKwargDataflowEdge<std::string, int> {
      return OpenKwargDataflowEdge<std::string, int>{
          KwargDataflowInputEdge<std::string, int>{
              /*src=*/src,
              /*dst=*/
              KwargDataflowInput<int>{
                  /*node=*/dst,
                  /*slot_name=*/dst_slot,
              },
          },
      };
    };

    auto mk_internal_edge = [](Node src, int src_slot, Node dst, int dst_slot)
        -> OpenKwargDataflowEdge<std::string, int> {
      return OpenKwargDataflowEdge<std::string, int>{
          KwargDataflowEdge<int>{
              /*src=*/KwargDataflowOutput<int>{
                  /*node=*/src,
                  /*slot_name=*/src_slot,
              },
              /*dst=*/
              KwargDataflowInput<int>{
                  /*node=*/dst,
                  /*slot_name=*/dst_slot,
              },
          },
      };
    };

    KwargDataflowOutput n1_0 = KwargDataflowOutput{n1, 0};
    KwargDataflowOutput n1_3 = KwargDataflowOutput{n1, 3};
    KwargDataflowOutput n1_4 = KwargDataflowOutput{n1, 4};
    KwargDataflowOutput n2_3 = KwargDataflowOutput{n2, 3};
    KwargDataflowOutput n3_1 = KwargDataflowOutput{n3, 1};
    KwargDataflowOutput n5_0 = KwargDataflowOutput{n5, 0};
    KwargDataflowOutput n5_4 = KwargDataflowOutput{n5, 4};

    OpenKwargDataflowEdge<std::string, int> e_n1_0_n2_1 =
        mk_internal_edge(n1, 0, n2, 1);
    OpenKwargDataflowEdge<std::string, int> e_n1_0_n3_0 =
        mk_internal_edge(n1, 0, n3, 0);
    OpenKwargDataflowEdge<std::string, int> e_n2_3_n4_1 =
        mk_internal_edge(n2, 3, n4, 1);
    OpenKwargDataflowEdge<std::string, int> e_n3_1_n5_1 =
        mk_internal_edge(n3, 1, n5, 1);

    KwargDataflowGraphInput<std::string> i1 =
        KwargDataflowGraphInput<std::string>{
            /*name=*/"a",
        };
    KwargDataflowGraphInput<std::string> i2 =
        KwargDataflowGraphInput<std::string>{
            /*name=*/"b",
        };

    OpenKwargDataflowEdge<std::string, int> e_i1_n1_3 =
        mk_input_edge(i1, n1, 3);
    OpenKwargDataflowEdge<std::string, int> e_i1_n1_4 =
        mk_input_edge(i1, n1, 4);
    OpenKwargDataflowEdge<std::string, int> e_i2_n5_4 =
        mk_input_edge(i2, n5, 4);

    OpenKwargDataflowGraphData<std::string, int> input =
        OpenKwargDataflowGraphData<std::string, int>{
            /*nodes=*/{n1, n2, n3, n4, n5},
            /*edges=*/
            {
                e_n1_0_n2_1,
                e_n1_0_n3_0,
                e_n2_3_n4_1,
                e_n3_1_n5_1,
                e_i1_n1_3,
                e_i1_n1_4,
                e_i2_n5_4,
            },
            /*inputs=*/
            {
                i1,
                i2,
            },
            /*outputs=*/
            {
                n1_0,
                n1_3,
                n1_4,
                n2_3,
                n3_1,
                n5_0,
                n5_4,
            }};

    OpenKwargDataflowGraphView<std::string, int> result =
        view_from_open_kwarg_dataflow_graph_data(input);
    OpenKwargDataflowGraphData<std::string, int> result_data =
        get_open_kwarg_dataflow_graph_data(result);

    OpenKwargDataflowGraphData<std::string, int> correct_data = input;

    CHECK(result_data == correct_data);
  }
}
