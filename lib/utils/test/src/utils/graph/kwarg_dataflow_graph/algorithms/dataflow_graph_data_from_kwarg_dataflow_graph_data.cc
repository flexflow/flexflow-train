#include "utils/graph/kwarg_dataflow_graph/algorithms/dataflow_graph_data_from_kwarg_dataflow_graph_data.h"
#include "utils/containers/reversed.h"
#include "utils/containers/sorted.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dataflow_graph_data_from_kwarg_dataflow_graph_data") {
    Node n0 = Node{0};
    Node n1 = Node{1};
    Node n2 = Node{2};

    KwargDataflowOutput<std::string> o0 = KwargDataflowOutput<std::string>{
        /*node=*/n0,
        /*slot_name=*/"a",
    };

    KwargDataflowOutput<std::string> o1 = KwargDataflowOutput<std::string>{
        /*node=*/n1,
        /*slot_name=*/"b",
    };

    KwargDataflowOutput<std::string> o2 = KwargDataflowOutput<std::string>{
        /*node=*/n1,
        /*slot_name=*/"c",
    };

    KwargDataflowOutput<std::string> o3 = KwargDataflowOutput<std::string>{
        /*node=*/n2,
        /*slot_name=*/"d",
    };

    auto mk_kwarg_edge =
        [](KwargDataflowOutput<std::string> const &src,
           Node dst_node,
           std::string dst_slot) -> KwargDataflowEdge<std::string> {
      return KwargDataflowEdge<std::string>{
          src,
          KwargDataflowInput<std::string>{
              dst_node,
              dst_slot,
          },
      };
    };

    KwargDataflowGraphData<std::string> input =
        KwargDataflowGraphData<std::string>{
            /*nodes=*/{
                n0,
                n1,
                n2,
            },
            /*edges=*/
            {
                mk_kwarg_edge(o1, n2, "z"),
                mk_kwarg_edge(o2, n2, "y"),
                mk_kwarg_edge(o0, n2, "x"),
            },
            /*outputs=*/
            {
                o0,
                o1,
                o2,
                o3,
            },
        };

    std::function<std::vector<std::string>(
        std::unordered_set<std::string> const &)>
        slot_ordering = [](std::unordered_set<std::string> const &slots)
        -> std::vector<std::string> { return reversed(sorted(slots)); };

    DataflowGraphData result =
        dataflow_graph_data_from_kwarg_dataflow_graph_data(input,
                                                           slot_ordering);

    DataflowGraphData correct = [&]() {
      DataflowOutput correct_o0 = DataflowOutput{
          /*node=*/n0,
          /*idx=*/0_n,
      };

      DataflowOutput correct_o1 = DataflowOutput{
          /*node=*/n1,
          /*idx=*/1_n,
      };

      DataflowOutput correct_o2 = DataflowOutput{
          /*node=*/n1,
          /*idx=*/0_n,
      };

      DataflowOutput correct_o3 = DataflowOutput{
          /*node=*/n2,
          /*idx=*/0_n,
      };

      auto mk_edge = [](DataflowOutput const &src,
                        Node dst_node,
                        nonnegative_int dst_idx) -> DataflowEdge {
        return DataflowEdge{
            src,
            DataflowInput{
                dst_node,
                dst_idx,
            },
        };
      };

      return DataflowGraphData{
          /*nodes=*/{
              n0,
              n1,
              n2,
          },
          /*edges=*/
          {
              mk_edge(correct_o0, n2, 2_n),
              mk_edge(correct_o1, n2, 0_n),
              mk_edge(correct_o2, n2, 1_n),
          },
          /*outputs=*/
          {
              correct_o0,
              correct_o1,
              correct_o2,
              correct_o3,
          },
      };
    }();

    CHECK(result == correct);
  }
}
