#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_subgraph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/view_from_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_data.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_kwarg_dataflow_graph_subgraph") {
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

    KwargDataflowGraphData<int> g_data =
        KwargDataflowGraphData<int>{/*nodes=*/{n1, n2, n3, n4, n5},
                                    /*edges=*/
                                    {
                                        mk_edge(n1, 0, n2, 1),
                                        mk_edge(n1, 0, n3, 0),
                                        mk_edge(n2, 3, n4, 1),
                                        mk_edge(n3, 1, n5, 1),
                                    },
                                    /*outputs=*/
                                    {
                                        KwargDataflowOutput{n1, 0},
                                        KwargDataflowOutput{n2, 3},
                                        KwargDataflowOutput{n3, 1},
                                        KwargDataflowOutput{n5, 0},
                                    }};

    KwargDataflowGraphView<int> g = view_from_kwarg_dataflow_graph_data(g_data);

    SUBCASE("node set is contains all graph nodes") {
      KwargDataflowGraphView<int> result = get_kwarg_dataflow_graph_subgraph(
          g, std::unordered_set{n1, n2, n3, n4, n5});
      KwargDataflowGraphData<int> result_data =
          get_kwarg_dataflow_graph_data(result);

      KwargDataflowGraphData<int> correct_data = g_data;

      CHECK(result_data == correct_data);
    }

    SUBCASE("node set is overlapping") {
      KwargDataflowGraphView<int> result =
          get_kwarg_dataflow_graph_subgraph(g, std::unordered_set{n2, n3, n5});
      KwargDataflowGraphData<int> result_data =
          get_kwarg_dataflow_graph_data(result);

      KwargDataflowGraphData<int> correct_data = KwargDataflowGraphData<int>{
          /*nodes=*/{n2, n3, n5},
          /*edges=*/
          {
              mk_edge(n3, 1, n5, 1),
          },
          /*outputs=*/
          {
              KwargDataflowOutput{n2, 3},
              KwargDataflowOutput{n3, 1},
              KwargDataflowOutput{n5, 0},
          },
      };

      CHECK(result_data == correct_data);
    }

    SUBCASE("node set is non-overlapping") {
      KwargDataflowGraphView<int> result =
          get_kwarg_dataflow_graph_subgraph(g, std::unordered_set<Node>{});
      KwargDataflowGraphData<int> result_data =
          get_kwarg_dataflow_graph_data(result);

      KwargDataflowGraphData<int> correct_data = KwargDataflowGraphData<int>{
          /*nodes=*/std::unordered_set<Node>{},
          /*edges=*/std::unordered_set<KwargDataflowEdge<int>>{},
          /*outputs=*/std::unordered_set<KwargDataflowOutput<int>>{},
      };

      CHECK(result_data == correct_data);
    }
  }
}
